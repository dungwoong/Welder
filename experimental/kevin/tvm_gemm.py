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

def sche_gemm(sch: tvm.tir.Schedule):
    C = sch.get_block("output")

    block_size_M, block_size_N = 256, 128 # block and warp tiling
    warp_size_M, warp_size_N = 128, 64
    chunk_size = 32 # size of each chunk in the k dimension
    warp_size = 32 # num threads in warp
    num_warp = (block_size_M * block_size_N) // (warp_size_M * warp_size_N)

    ax_M, ax_N, ax_K = sch.get_loops(C) # find MNK
    grid_M, tm, block_M = sch.split(ax_M, factors=[None, 4, block_size_M]) # we can see we split ax_M into [None, 4, block_size_M] so we have 3 loops, and the last one is parallelized in the grid
    grid_N, tn, block_N = sch.split(ax_N, factors=[None, 8, block_size_N])
    sch.reorder(grid_M, grid_N, tm, tn, block_M, block_N) # we reorder the loops like so
    grid = sch.fuse(grid_M, grid_N, tm, tn) # grid will iterate over all combinations of these things
    sch.bind(grid, "blockIdx.x") # grid is bound to blockIdx.x now. tm, tn gets us fine-grained scheduling kinda

    grid, ax_M, ax_N, ax_K = sch.get_loops(C)
    K_outer, K_inner = sch.split(ax_K, factors=[None, chunk_size]) # so e.g. loop over chunks of 32
    warp_M, inner_M = sch.split(ax_M, factors=[None, warp_size_M]) # split ax_M and ax_N by the warp size
    warp_N, inner_N = sch.split(ax_N, factors=[None, warp_size_N])
    sch.reorder(warp_M, warp_N, K_outer, inner_M, inner_N, K_inner) # reorder so that the inner loop processes the BM BN BK tiles
    warp = sch.fuse(warp_M, warp_N) # different warps will process different blocks of these things, each warp processes k_outer in their loop
    sch.bind(warp, "threadIdx.y") # bind warp to threadIdx.y

    # layout in smem I guess
    layoutA = RowMajorTensorOpMultiplicandCrosswise(block_size_M, chunk_size)
    layoutB = RowMajorTensorOpMultiplicandCongruous(chunk_size, block_size_N)

    for idx in [0, 1]:
        layout = layoutA if idx==0 else layoutB
        SS = sch.cache_read(C, idx, "shared") # cache either A or B into SMEM (the idx'th input of C)
        sch.compute_at(SS, K_outer) # place the loads INSIDE the loop over K_outer, so only load once per loop
        
        # pad the layout if we need to, so we are compatible with tensor core requirements
        if layout.requires_padding():
            pad_size = 4 if idx == 0 else 8 # m8n8k4
            layout.set_pad(pad_size)
            sch.storage_align(SS, 0, axis=-2, factor=32, offset=pad_size) # ???
        
        # I think this part is just reorganizing all the loops over SS into shapes that we can do with wmma?
        fused = sch.fuse(*sch.get_loops(SS)[-2:])
        vectorize_size = layout.get_vectorize()
        oo, idx_y, idx_x, vec = sch.split(fused, [None, num_warp, warp_size, vectorize_size])
        sch.bind(idx_x, "threadIdx.x")
        sch.bind(idx_y, "threadIdx.y")
        sch.vectorize(vec)
        # sch.unroll(oo)

    cls_code = register_cutlass_warp_mma(warp_size_M, warp_size_N, chunk_size, layoutA, layoutB)
    C_warp = sch.cache_write(C, 0, "cutlass.warp.mma") # declare a buffer for writing results of mma
    sch.reverse_compute_at(C_warp, warp) # moves computation of C_warp to happen AFTER the warp_level loop(??!)

    sch.decompose_reduction(C, sch.get_loops(C)[2]) # breaks reduction block into C_init, C_update (C=0 and C += ...), so we tensorize the zeroing of C separate from accumulation
    block_init_c = sch.get_block("output_init")
    layoutC = FragmentCLayout8x8(warp_size_M, warp_size_N)

    sch.transform_loop(C_warp, 2, layoutC)
    sch.bind(sch.get_loops(C_warp)[-2], "threadIdx.x")
    oo, vec = sch.split(sch.get_loops(C_warp)[-1], factors=[None, layoutC.get_vectorize()])
    sch.vectorize(vec)
    sch.annotate(oo, "pragma_unroll_explicit", False)
    sch.unroll(oo)


    sch.annotate(sch.get_loops(C)[2], "software_pipeline_stage", [0, 0, 1, 1, 2])
    sch.annotate(sch.get_loops(C)[2], "software_pipeline_order", [0, 1, 2, 4, 3])
    sch.annotate(sch.get_loops(C)[2], "software_pipeline_async_stages", [0])
    sch.tensorize(sch.get_loops(block_init_c)[-2],
        register_cutlass_warp_init_intrin(warp_size_M, warp_size_N, "float16",
        cls_code, block_size_M // warp_size_M, block_size_N // warp_size_N)
    )
    sch.tensorize(sch.get_loops(C)[-3],
        register_gemm_intrin(
            warp_size_M, warp_size_N, chunk_size, "float16", "float16", False, False, layoutA, layoutB)
    )
    layout_pass = ApplyLayoutPass({"a_shared": layoutA, "b_shared": layoutB, "output_cutlass.warp.mma": layoutC.fragment_offset})
    passes = [
        layout_pass.get_pass(),
        (3, tvm.tir.transform.InjectPTXAsyncCopy()),
    ]
    # print(sch.mod["main"].script())
    # exit(0)

    grid = [np.prod(args[-1].shape) // block_size_M // block_size_N, 1, 1]
    block = [warp_size, num_warp, 1]
    return grid, block, passes


args = gemm(4096, 4096, 4096)
workload = te.create_prim_func(args) # lower tensor expressions into a TVM TIR prim func
ir_module = tvm.IRModule({"main": workload}) # wraps PrimFunc in an IRModule so we can now print, schedule etc.
sch = tvm.tir.Schedule(ir_module)
from welder.IRpass import *

grid, block, passes = sche_gemm(sch)
with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}):
    mod = tvm.build(sch.mod["main"], target="cuda")
kernel_code = mod.imported_modules[0].get_source()
kernel_code = kernel_code[kernel_code.index('extern "C" __global__ void'):]

# print(kernel_code)
cp = CompileResult(None, kernel_code, block, grid, "default_function_kernel", args)
cp.compile_and_load(welder.arch.cuda())
a = cp.get_example_outputs()[0]
print(a)
print(cp.profile())

# from welder.reference import get_reference_output

# oo = get_reference_output(args)[-1].numpy()
# print(oo)
# print(abs(oo - a).max())
