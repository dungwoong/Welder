// idk blocksize or gridsize yet
// configs[0]['globals']={'Rasterization': <Rasterization2DRow(12)>}
// {'block': [128, 128], 'warp': [64, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '86', 'strides': {2: <Stride, 0, 136>}}
// idk where the wmma 16, 8, 16 goes but I'll just assume it is inside the cutlass objects. Anyways, we can see their strat now.

// blockdim 32, 4, 1
// griddim is 1024

// so each block does 128x128, each warp does 64x64 so there's 4 warps per block and you reduce over 32 at a time so [64x32]x[32x64] at a time I guess?

__global__ void __launch_bounds__(128) Group0(half *__restrict__ p0, half *__restrict__ p1, half *__restrict__ T_matmul_NN)
{
    int __bid = blockIdx.x;
    const dim3 blockIdx(rasterization2DRow<32, 32, 12>(__bid), 0, 0);
    __shared__ half p0_shared[8192]; // 64x32 x 2(k) x 2(buffer I guess)
    __shared__ half p1_shared[8192]; // 64x32 x 2(k) x 2(buffer I guess)
    ALLOCATE_CUTLASS_OBJECT(T_matmul_NN_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
                                                              cutlass::gemm::GemmShape<64, 64, 32>, // [64x32]x[32x64]
                                                              cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
                                                              cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>>((((int)threadIdx.y) >> 1), (((int)threadIdx.y) & 1), ((int)threadIdx.x)));

    // FETCH A 32x128
    // (fetches 4 x 8 items --> 32x128 for the entire warp)
#pragma unroll
    for (int ax0_ax1_fused_0_0_0 = 0; ax0_ax1_fused_0_0_0 < 4; ++ax0_ax1_fused_0_0_0)
    {

        {
            unsigned int addr;
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(p0_shared + (((((ax0_ax1_fused_0_0_0 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(p0 + ((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_0_0 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16));
        }
    }

    // FETCH B 128x32
#pragma unroll
    for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 4; ++ax0_ax1_fused_0_0_0_1)
    {

        {
            unsigned int addr;
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(p1_shared + ((((((ax0_ax1_fused_0_0_0_1 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.y) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((int)threadIdx.y) & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(p1 + (((((ax0_ax1_fused_0_0_0_1 * 32768) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)))), "n"(16));
        }
    }
    __asm__ __volatile__("cp.async.commit_group;");


    // THE SECOND GROUP OF 32x128 FETCHES ============================================================================================================
    // FETCH A 32x128
#pragma unroll
    for (int ax0_ax1_fused_0_0_0_2 = 0; ax0_ax1_fused_0_0_0_2 < 4; ++ax0_ax1_fused_0_0_0_2)
    {

        {
            unsigned int addr;
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(p0_shared + ((((((ax0_ax1_fused_0_0_0_2 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096))));
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(p0 + (((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_0_0_2 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16));
        }
    }
    
    // FETCH B 32x128
#pragma unroll
    for (int ax0_ax1_fused_0_0_0_3 = 0; ax0_ax1_fused_0_0_0_3 < 4; ++ax0_ax1_fused_0_0_0_3)
    {

        {
            unsigned int addr;
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(p1_shared + (((((((ax0_ax1_fused_0_0_0_3 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.y) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((int)threadIdx.y) & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096))));
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(p1 + ((((((ax0_ax1_fused_0_0_0_3 * 32768) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072))), "n"(16));
        }
    }
    __asm__ __volatile__("cp.async.commit_group;");

    __asm__ __volatile__("cp.async.wait_group 1;");

    // the first mma(idk why it's out of the main loop, but yeah it's just a 2-stage m128n128k32 each warp takes m64n64k32)
    __syncthreads();
    call_cutlass_mma_prologue(T_matmul_NN_cutlass_warp_mma, (&(p0_shared[0])), (&(p1_shared[0])), 32, 128);
    call_cutlass_mma_body(T_matmul_NN_cutlass_warp_mma);
    for (int k_0 = 0; k_0 < 126; ++k_0)
    {
#pragma unroll
        for (int ax0_ax1_fused_0_0_0_4 = 0; ax0_ax1_fused_0_0_0_4 < 4; ++ax0_ax1_fused_0_0_0_4)
        {

            {
                unsigned int addr;
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(p0_shared + (((((((k_0 & 1) * 4096) + (ax0_ax1_fused_0_0_0_4 * 1024)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(p0 + ((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0_ax1_fused_0_0_0_4 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16));
            }
        }
#pragma unroll
        for (int ax0_ax1_fused_0_0_0_5 = 0; ax0_ax1_fused_0_0_0_5 < 4; ++ax0_ax1_fused_0_0_0_5)
        {

            {
                unsigned int addr;
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(p1_shared + ((((((((k_0 & 1) * 4096) + (ax0_ax1_fused_0_0_0_5 * 1024)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.y) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((int)threadIdx.y) & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(p1 + (((((((k_0 * 131072) + (ax0_ax1_fused_0_0_0_5 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144))), "n"(16));
            }
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 1;");

        __syncthreads();
        call_cutlass_mma_prologue(T_matmul_NN_cutlass_warp_mma, (&(p0_shared[(((k_0 + 1) & 1) * 4096)])), (&(p1_shared[(((k_0 + 1) & 1) * 4096)])), 32, 128);
        call_cutlass_mma_epilogue(T_matmul_NN_cutlass_warp_mma);
        call_cutlass_mma_body(T_matmul_NN_cutlass_warp_mma);
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    call_cutlass_mma_prologue(T_matmul_NN_cutlass_warp_mma, (&(p0_shared[4096])), (&(p1_shared[4096])), 32, 128);
    call_cutlass_mma_epilogue(T_matmul_NN_cutlass_warp_mma);
    call_cutlass_mma_body(T_matmul_NN_cutlass_warp_mma);
    call_cutlass_mma_epilogue(T_matmul_NN_cutlass_warp_mma);
// apparently uint1 is just like a uint so that's 4bytes = 2 halfs
// so 2 * 64 = 128, so 128 threads, each thread saves 128, that's 128 elements. ok cool
#pragma unroll
    for (int ax1_0 = 0; ax1_0 < 64; ++ax1_0)
    {
        *(uint1 *)(T_matmul_NN + (((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.y) >> 1) * 262144)) + ((ax1_0 & 7) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + ((ax1_0 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1 *)(T_matmul_NN_cutlass_warp_mma + (ax1_0 * 2));
    }
}