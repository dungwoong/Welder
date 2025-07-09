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
GOAL: add SMEM caching and have each warp do a 16x16 mma op
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

# https://tvm.apache.org/docs/reference/api/python/tir/schedule.html
# - cache_read makes a cache
def schedule(sch: tvm.tir.Schedule):
    C = sch.get_block("output") # we defined this block in te.compute above
    block_size_m, block_size_n = 16, 16
    i, j, k = sch.get_loops(C)
    i0, i1 = sch.split(i, factors=[None, block_size_m])
    j0, j1 = sch.split(j, factors=[None, block_size_n])
    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")

    grid = [np.ceil(MNK / block_size_m), np.ceil(MNK / block_size_n), 1]
    block = [block_size_m, block_size_n, 1]
    return grid, block

def schedule(sch: tvm.tir.Schedule):
    # following the tvm_gemm example
    C = sch.get_block("output")

    # let's do 2x2 wmma's so 32x32 per block, 16x16 per warp
    block_size_M, block_size_N = 32, 32
    warp_size_M, warp_size_N = 16, 16
    chunk_size = 16
    warp_size = 32 # n threads per warp
    num_warp = 4 # HARDCODED

    ax_m, ax_n, ax_k = sch.get_loops(C)
    grid_m, block_m = sch.split(ax_m, factors=[None, block_size_M])
    grid_n, block_n = sch.split(ax_n, factors=[None, block_size_N])
    sch.reorder(grid_m, grid_n, block_m, block_n)
    grid = sch.fuse(grid_m, grid_n)
    sch.bind(grid, 'blockIdx.x')

    grid, ax_m, ax_n, ax_k = sch.get_loops(C) # re-fetch loops since we fused grid
    k_outer, k_inner = sch.split(ax_k, factors=[None, chunk_size]) # for k_outer: load a k_inner tile
    warp_m, inner_m = sch.split(ax_m, factors=[None, warp_size_M]) # same idea, we later use wmma to tensorize our op
    warp_n, inner_n = sch.split(ax_n, factors=[None, warp_size_N])
    sch.reorder(warp_m, warp_n, k_outer, inner_m, inner_n, k_inner)
    warp = sch.fuse(warp_m, warp_n)
    sch.bind(warp, 'threadIdx.y') # so the warp m and n will be split among threadIdx.y

    for idx in [0, 1]:
        SS = sch.cache_read(C, idx, "shared") # read to shared
        sch.compute_at(SS, k_outer) # move producer block UNDER the specific loop
        sch.storage_align(SS, 0, axis=-2, factor=32, offset=0) # idk what this is, but we gotta use it. Should work since our size is 4096, but may not be good for other sizes
        fused = sch.fuse(*sch.get_loops(SS)[-2:]) # fuse loops
        
        # this is how we do the strided loading pattern where each thread loads 8 elements at a time to load the tile in
        vec = 8 # load 8 items at a time into smem
        o, ty, tx, v = sch.split(fused, factors=[None, num_warp, warp_size, vec])
        sch.bind(ty, 'threadIdx.y')
        sch.bind(tx, 'threadIdx.x')
        sch.vectorize(v)
    
    c_warp = sch.cache_write(C, 0, "wmma.accumulator")
    sch.reverse_compute_at(c_warp, warp) # move write into accumulator to AFTER warp does their thing

    from welder.schedule.wmma_intrin import (intrin_wmma_gemm, intrin_wmma_load_matrix_A,
                                  intrin_wmma_load_matrix_W,
                                  intrin_wmma_store_matrix)

# def schedule(sch: tvm.tir.Schedule, config) -> tuple:
#     # Get output block
#     C = sch.get_block("output")

#     # Cache read to shared memory
#     AS = sch.cache_read(C, 0, "shared")
#     BS = sch.cache_read(C, 1, "shared")

#     # Cache read to wmma matrix scopes
#     AF = sch.cache_read(AS, 0, "wmma.matrix_a")
#     BF = sch.cache_read(BS, 0, "wmma.matrix_b")

#     # Cache write to wmma accumulator
#     CF = sch.cache_write(C, 0, "wmma.accumulator")

#     # Get loops for C
#     i, j, k = sch.get_loops(C)
#     block_size_m, block_size_n, block_size_k = (16, 16, 16)  # wmma config

#     # Split loops according to WMMA tile sizes
#     i0, i1 = sch.split(i, factors=[None, block_size_m])
#     j0, j1 = sch.split(j, factors=[None, block_size_n])
#     k0, k1 = sch.split(k, factors=[None, block_size_k])

#     # Bind block and thread indices
#     sch.bind(i0, "blockIdx.y")
#     sch.bind(j0, "blockIdx.x")
#     sch.bind(i1, "threadIdx.y")
#     sch.bind(j1, "threadIdx.x")

#     # Move shared memory loads inside block loops
#     sch.compute_at(AS, k0)
#     sch.compute_at(BS, k0)

#     # Move wmma matrix loads inside the inner reduction loop
#     sch.compute_at(AF, k1)
#     sch.compute_at(BF, k1)

#     # Compute CF at the right scope (inside warp loops)
#     sch.compute_at(CF, k1)

#     # TODO: tensorize AF, BF, CF with WMMA intrinsics here (requires registered intrinsics)

#     # Return launch grid/block size (ceil div of your problem size)
#     grid = [int(np.ceil(MNK / block_size_m)), int(np.ceil(MNK / block_size_n)), 1]
#     block = [block_size_m, block_size_n, 1]

#     return grid, block

args = gemm(MNK, MNK, MNK)
workload = te.create_prim_func(args)
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)

print_if_y('Print initial TIR[y/n]', sch.mod["main"].script()) # schedule has modules and we get the main one and get the script
grid, block = schedule(sch)

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


latency = lib.profile(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
print(f'Latency from lib.profile: {latency}')

c_actual = torch_arrs[-1]
c_ref = torch_arrs[0] @ torch_arrs[1]

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