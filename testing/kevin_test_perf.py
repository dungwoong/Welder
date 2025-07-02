import argparse
import os
import tvm

import numpy as np
import onnx
import onnxruntime as ort
from tvm.contrib import graph_executor
from tvm.contrib.debugger.debug_runtime import debug_executor
import time

def get_max_diff(tensor_list_a, tensor_list_b):
    assert len(tensor_list_a) > 0
    total_diff = [0]
    for a, b in zip(tensor_list_a, tensor_list_b):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        if np.any(np.logical_and(np.isnan(a), np.logical_not(np.isnan(b)))):
            return 1e7
        assert a.shape == b.shape
        diff = np.abs(a-b)
        diff /= np.abs(b).clip(1) # handle large floating numbers
        diff = np.max(diff)
        total_diff.append(diff)
    total_diff = max(total_diff)
    return total_diff

def ref_output(onnx_model_path):
    np.random.seed(0)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_inputs = {}
    inputs = []
    for value in ort_session.get_inputs():
        if value.type == 'tensor(int64)':
            tensor = np.ones(value.shape).astype(np.int64)
        elif value.type == 'tensor(float16)':
            tensor = np.random.normal(size=value.shape).astype(np.float16)
        elif value.type == 'tensor(float)':
            tensor = np.random.normal(size=value.shape).astype(np.float32)
        else:
            raise NotImplementedError(value.type)
        ort_inputs[value.name] = tensor
        inputs.append(tensor)
    outputs = ort_session.get_outputs()
    outputs_name = [item.name for item in outputs]
    outputs = ort_session.run(outputs_name, ort_inputs)
    return inputs, outputs

def setup_welder(prefix, inputs):
    import tvm
    lib_path = os.path.join(prefix, "model.so")
    lib = tvm.runtime.load_module(lib_path)
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cuda(0)))
    for i, tensor in enumerate(inputs):
        rt_mod.set_input(i, tensor)
    return rt_mod

def get_welder_outs(prefix, inputs):
    import tvm
    lib_path = os.path.join(prefix, "model.so")
    lib = tvm.runtime.load_module(lib_path)
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cuda(0)))
    for i, tensor in enumerate(inputs):
        rt_mod.set_input(i, tensor)
    rt_mod.run()
    outputs = []
    for i in range(rt_mod.get_num_outputs()):
        out = rt_mod.get_output(i).asnumpy()
        outputs.append(out)
    return outputs

def get_self_attn_flops(b=16, n=1024, h=16, d=64):
    b, n, h, d = 16, 1024, 16, 64
    return ((2 * b * h * n * n* d) + (4 * b * h * n * n) + (2 * b * h * n * n * d))

def get_gemm_flops(dim=4096):
    return 2 * dim * dim * dim

def get_batched_gemm_flops(dim=4096, L=4):
    return L * get_gemm_flops(dim)

def get_dual_gemm_flops(dim=4096):
    return 2 * get_gemm_flops(dim) + (dim * dim) # for accumulation

def get_gemm_and_reduction_flops(dim=4096):
    return get_gemm_flops(dim) + (dim * dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()

    prefix = args.prefix
    inputs, outputs_ref = ref_output(os.path.join(prefix, "model.onnx"))

    rt_mod = setup_welder(prefix, inputs)
    iters = 1 # I think after 1 run it just caches stuff, or maybe it's a generator or smth
    rt_mod.run()
    
    bench = rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False)
    # print(f'{bench.max=}, {bench.median=}, {bench.std=}, {bench.min=}, {bench.max=}')
    print(bench)
    median_s = bench.median

    flops = get_self_attn_flops()
    print(f'Median time: {round(1e3 * median_s, 3)} ms')
    print(f'Median GFLOPs: {round((flops / median_s) * 1e-9, 2)}')

    outputs = []
    for i in range(rt_mod.get_num_outputs()):
        out = rt_mod.get_output(i).asnumpy()
        outputs.append(out)

    print('Correctness checking...')
    max_diff = get_max_diff(outputs, outputs_ref)

    # NOTE: problematic since outputs could be a list of tensors or whatever.
    # outputs = np.array(outputs).flatten()
    # outputs_ref = np.array(outputs_ref).flatten()
    # print(outputs[-8:])
    # print(outputs_ref[-8:])
    print("Output max diff : ", max_diff)

