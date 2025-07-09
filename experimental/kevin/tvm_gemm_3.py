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
GOAL: have each warp do a 16x16 op with wmma
"""

MNK=4096

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

def schedule(sch: tvm.tir.Schedule):
    # following the tvm_gemm example
    wmma_m, wmma_n, wmma_k = 16, 16, 16
    C = sch.get_block("output")

    # let's do 32x32 items per block, 1x1 per thread
    block_size = (32, 32, 32)
    warp_size = (16, 16, 32)
    warps_per_block = (2, 2, 2) # HARDCODED
    mma_size = (16, 16, 16)
    num_warp = 4 # HARDCODED
    threads_per_block = num_warp * 32

    ax_m, ax_n, ax_k = sch.get_loops(C)
    # read these right to left so 16, x2=32, rest is in the loop
    grid_m, warp_m, mma_m = sch.split(ax_m, factors=[None, warps_per_block[0], mma_size[0]]) # normally one warp would do multiple mmas but each warp does 1 for now.
    grid_n, warp_n, mma_n = sch.split(ax_n, factors=[None, warps_per_block[1], mma_size[1]])
    main_loop_k, warp_loop_k, mma_k = sch.split(ax_k, factors=[None, warps_per_block[2], mma_size[2]])
    sch.reorder(grid_m, grid_n, warp_m, warp_n, main_loop_k, warp_loop_k, mma_k, mma_m, mma_n)
    
    # grid vars
    sch.bind(grid_m, 'blockIdx.y')
    sch.bind(grid_n, 'blockIdx.x')

    # warp var
    warp = sch.fuse(warp_m, warp_n)
    sch.bind(warp, 'threadIdx.y')

    # -----------------
    # read to shared
    # -----------------
    AS = sch.cache_read(C, 0, "shared")
    BS = sch.cache_read(C, 1, "shared")
    sch.compute_at(AS, main_loop_k)
    sch.compute_at(BS, main_loop_k)

    # cooperative loading
    fused = sch.fuse(*sch.get_loops(AS)[-2:])
    oo, idx_a = sch.split(fused, [None, threads_per_block])
    sch.bind(idx_a, 'threadIdx.x')
    fused = sch.fuse(*sch.get_loops(BS)[-2:])
    oo, idx_b = sch.split(fused, [None, threads_per_block])
    sch.bind(idx_b, 'threadIdx.x')

    # read to reg
    CF = sch.cache_write(C, 0, "wmma.accumulator")
    AR = sch.cache_read(C, 0, "wmma.matrix_a")
    BR = sch.cache_read(C, 1, "wmma.matrix_b")
    sch.compute_at(AR, warp_loop_k)
    sch.compute_at(BR, warp_loop_k)
    sch.reverse_compute_at(CF, warp)
    # let's hardcode strides for now
    
    # grid_dim = [MNK//block_size_M, MNK//block_size_N, 1]
    # block_dim = [block_size_M * block_size_N, 1, 1]
    # return grid_dim, block_dim
    return None, None


args = gemm(MNK, MNK, MNK)
workload = te.create_prim_func(args)
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)

print_if_y('Print initial TIR[y/n]', sch.mod["main"].script()) # schedule has modules and we get the main one and get the script
grid, block = schedule(sch)
print_if_y('Print scheduled TIR[y/n]', sch.mod["main"].script()) # schedule has modules and we get the main one and get the script

# https://tvm.apache.org/docs/reference/api/python/tir/schedule.html
# from welder.IRpass import *

mod = tvm.build(sch.mod["main"], target="cuda")
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
max_err_idx = tuple(i[0].item() for i in max_err_idx) # if there's multiple
ref_val = c_ref[max_err_idx].item()
actual_val = c_actual[max_err_idx].item()
rel_err_at_max = abs(actual_val - ref_val) / (abs(ref_val) + 1e-6)

print(f'{max_err=} {mean_err=} {rel_err_at_max=}') # note that when we use half instead of float the error becomes really large like max error = 8
