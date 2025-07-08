import numpy as np
import welder
import tvm
from tvm import te
from welder.layout import *
from welder.schedule.cutlass_intrin import *
from welder.utils import CompileResult

from welder.tvm_build import unset_tvm_cuda_compile
unset_tvm_cuda_compile()

def gemm(n, m, k):
    """TVM expression for vector add"""
    A = te.placeholder((n, k), dtype="float16", name='a')
    B = te.placeholder((k, m), dtype="float16", name='b')
    K = te.reduce_axis((0, k))
    C = te.compute((n, m), lambda i, j: te.sum(A[i,K]*B[K,j], axis=[K]), name='output')
    return A, B, C

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

    grid = [np.ceil(4096 / block_size_m), np.ceil(4096 / block_size_n), 1]
    block = [block_size_m, block_size_n, 1]
    return grid, block

args = gemm(4096, 4096, 4096)
workload = te.create_prim_func(args)
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)

print(sch.mod["main"].script()) # schedule has modules and we get the main one and get the script
input('Press anything to continue')
grid, block = schedule(sch)

# https://tvm.apache.org/docs/reference/api/python/tir/schedule.html
# from welder.IRpass import *

mod = tvm.build(sch.mod["main"], target="cuda")
kernel_code = mod.imported_modules[0].get_source()
kernel_code = kernel_code[kernel_code.index('extern "C" __global__ void'):]
print(kernel_code)
print(f'Kernel Code Generated')

cp = CompileResult(None, kernel_code, block, grid, 'default_function_kernel', args) # you need to call it default_function_kernel because tvm names it like that
lib = cp.compile_and_load(welder.arch.cuda())

# ---------------------------
# Profiling
# ---------------------------

a_tvm, b_tvm, c_tvm = args
m, k = a_tvm.shape # I use mnk, their code used nmk
_, n = b_tvm.shape

# create numpy inputs
a_np = np.random.rand(*a_tvm.shape).astype("float16")
b_np = np.random.rand(*b_tvm.shape).astype("float16")
c_np = np.zeros((n, m), dtype="float16")  # output placeholder