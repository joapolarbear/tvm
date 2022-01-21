

from tvm import autotvm

from os
import time
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

target = "cuda"

### Load ONNX model
model_path = "resnet50-v2-7.onnx"
onnx_model = onnx.load(model_path)

### Load Input data
batch_size = 1
img_data = np.ones((batch_size, 3, 224, 224))
input_name = "data"
output_shape = (batch_size, 1000)
shape_dict = {input_name: img_data.shape}

### Convert ONNX model to Relay IR Module
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

runner = autotvm.LocalRunner(
        number=10,
        repeat=1,
        timeout=10,
        min_repeat_ms=0,
        enable_cpu_cache_flush=True,
)
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

### for each task
i = 0
task = tasks[i]
prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
tuner_obj = autotvm.tuner.XGBTuner(task, loss_type="rank")

### tuner_obj.tune(...)
n_trial = 10
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(build_func="default"), runner=runner
)
measure_batch = autotvm.measure.create_measure_batch(tuner_obj.task, measure_option)
n_parallel = getattr(measure_batch, "n_parallel", 1)


### for each trial
configs = tuner_obj.next_batch(min(n_parallel, n_trial - i))
inputs = [autotvm.measure.MeasureInput(tuner_obj.task.target, tuner_obj.task, config) for config in configs]
results = measure_batch(inputs)

# for each pair of input and result
inp = inputs[0]
res = results[0]
config = inp.config
flops = inp.task.flop / np.mean(res.costs)

xs = []
ys = []
index = inp.config.index
xs.append(index)
ys.append(flops)


### in cost model fit
tic = time.time()

tuner_obj.cost_model._reset_pool(tuner_obj.cost_model.space, tuner_obj.cost_model.target, tuner_obj.cost_model.task)
x_train = tuner_obj.cost_model._get_feature(xs)

xgb = __import__("xgboost")
type(xgb)

y_train = np.array(ys)
y_max = np.max(y_train)
y_train = y_train / max(y_max, 1e-8)

valid_index = y_train > 1e-6
index = np.random.permutation(len(x_train))
dtrain = xgb.DMatrix(x_train[index], y_train[index])
tuner_obj.cost_model._sample_size = len(x_train)

plan_size=64
plan_size *= 2
bst = xgb.train(
    tuner_obj.cost_model.xgb_params,
    dtrain,
    num_boost_round=8000,
    callbacks=[
        autotvm.tuner.xgboost_cost_model.custom_callback(
            stopping_rounds=20,
            metric="tr-a-recall@%d" % plan_size,
            evals=[(dtrain, "tr")],
            maximize=True,
            fevals=[
                autotvm.tuner.xgboost_cost_model.xgb_average_recalln_curve_score(plan_size),
            ],
            verbose_eval=tuner_obj.cost_model.log_interval,
        )
    ],
)

import logging
logger = logging.getLogger("autotvm")
logger.setLevel("INFO")
logger.info(
    "XGB train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
    time.time() - tic,
    len(xs),
    len(xs) - np.sum(valid_index),
    tuner_obj.cost_model.feature_cache.size(tuner_obj.cost_model.fea_type),
)