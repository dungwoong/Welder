import numpy as np
import welder
import tvm
from tvm import te
from welder.layout import *
from welder.schedule.cutlass_intrin import *
from welder.utils import CompileResult
import torch
import ctypes

from welder.tvm_build import unset_tvm_cuda_compile
unset_tvm_cuda_compile()

"""
GOAL: add SMEM caching and have each warp do a 16x16 op(no mma yet)
"""

MNK=128

def print_if_y(message, s):
    x = input(message)
    if x == 'y':
        print(s)

def gemm(n, m, k):
    """TVM expression for GEMM"""
    A = te.placeholder((n, k), dtype="float16", name='a')
    B = te.placeholder((k, m), dtype="float16", name='b')
    K = te.reduce_axis((0, k))
    C = te.compute((n, m), lambda i, j: te.sum(A[i,K]*B[K,j], axis=[K]), name='output')
    return A, B, C

def schedule(sch: tvm.te.Schedule):
    C = sch.outputs[0]
    A, B = C.input_tensors
    # print(A, B, C)
    i, j, k = C.axis

    warp_size = (32, 32)
    grid_i, thread_i = sch.split()
    grid_j, thread_j

    sch[C].bind(i, te.thread_axis("blockIdx.y"))
    sch[C].bind(j, te.thread_axis(""))
    pass


args = gemm(MNK, MNK, MNK)
A, B, C = gemm(MNK, MNK, MNK)
sch = te.create_schedule(C.op)

print_if_y('Print initial TIR[y/n]', tvm.lower(sch, [A, B, C])) # schedule has modules and we get the main one and get the script
schedule(sch)
print_if_y('Print scheduled TIR[y/n]', tvm.lower(sch, [A, B, C])) # schedule has modules and we get the main one and get the script

# https://tvm.apache.org/docs/reference/api/python/tir/schedule.html
# from welder.IRpass import *

ir_module = tvm.lower(sch, [A, B, C], name="main")
mod = tvm.build(ir_module, target="cuda")
kernel_code = mod.imported_modules[0].get_source()
kernel_code = kernel_code[kernel_code.index('extern "C" __global__ void'):]
print_if_y('Print kernel code? [y/n]', kernel_code)
print(f'Kernel Code Generated')

# ---------------------------
# Profiling
# ---------------------------

cp = CompileResult(None, kernel_code, block, grid, 'default_function_kernel', args) # you need to call it default_function_kernel because tvm names it like that
lib = cp.compile_and_load(welder.arch.cuda())
print('Latency from CompileResult.profile:', cp.profile())
torch_arrs = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device=}')
for arg in args:
    shape = tuple(map(int, arg.shape))
    arr = torch.randn(shape, device=device, dtype=torch.float16)
    torch_arrs.append(arr)


print(f'{type(lib)=}')
latency = lib.profile(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs]) # look at compileresult, it attaches a .profile function, that's what it's running. This is in ms.
print(f'Latency from lib.profile: {latency}')

flops = 2 * 4096 * 4096 * 4096
print(f'GFLOPS: {(flops / (latency * 1e-3)) * 1e-9}')

c_actual = torch_arrs[-1]
c_ref = torch_arrs[0] @ torch_arrs[1]
# print(f'{c_actual.shape=}, {c_ref.shape=}')

abs_error = (c_actual - c_ref).abs()
max_err = abs_error.max().item()
mean_err = abs_error.abs().mean().item()
max_err_idx = (abs_error == abs_error.max()).nonzero(as_tuple=True)
ref_val = c_ref[max_err_idx].item()
actual_val = c_actual[max_err_idx].item()
rel_err_at_max = abs(actual_val - ref_val) / (abs(ref_val) + 1e-6)

print(f'{max_err=} {mean_err=} {rel_err_at_max=}') # note that when we use half instead of float the error becomes really large like max error = 8

# this is how Welder does profiling, doesn't work here bcz no profile impl added
# from welder.arch.cuda import cuda
# arch = cuda()
# p = PopenPoolExecutor(max_workers=1, timeout=None, initializer=profiler.init_server, initargs=[arch])
# future = p.submit(profiler.call_profile, cp.lib_name, cp.args, dev)
# print(future.result())