__device__ void Group2_0_welder_matmul_2(half *__restrict__ p0, half *__restrict__ p1, half *__restrict__ T_matmul, char *shared)
{
    int __flatten_tid = threadIdx.x;
    const dim3 threadIdx(__flatten_tid % 32, __flatten_tid / 32, 0);
    half *p0_shared = (half *)(shared + 0);
    half *p1_shared = (half *)(shared + 16384);
    ALLOCATE_CUTLASS_OBJECT(T_matmul_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
                                                           cutlass::gemm::GemmShape<64, 32, 32>,
                                                           cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
                                                           cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>>((((int)threadIdx.y) >> 1), (((int)threadIdx.y) & 1), ((int)threadIdx.x)));
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
                "l"((void *)(p0 + (((((((int)blockIdx.x) * 131072) + (ax0_ax1_fused_0_0_0 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16));
        }
    }
#pragma unroll
    for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 2; ++ax0_ax1_fused_0_0_0_1)
    {

        {
            unsigned int addr;
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(p1_shared + ((((((ax0_ax1_fused_0_0_0_1 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.y) & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(p1 + (((((((int)blockIdx.x) >> 3) * 65536) + (ax0_ax1_fused_0_0_0_1 * 1024)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)))), "n"(16));
        }
    }
    __asm__ __volatile__("cp.async.commit_group;");

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
                "l"((void *)(p0 + ((((((((int)blockIdx.x) * 131072) + (ax0_ax1_fused_0_0_0_2 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16));
        }
    }
#pragma unroll
    for (int ax0_ax1_fused_0_0_0_3 = 0; ax0_ax1_fused_0_0_0_3 < 2; ++ax0_ax1_fused_0_0_0_3)
    {

        {
            unsigned int addr;
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(p1_shared + (((((((ax0_ax1_fused_0_0_0_3 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.y) & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048))));
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(p1 + ((((((((int)blockIdx.x) >> 3) * 65536) + (ax0_ax1_fused_0_0_0_3 * 1024)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 2048))), "n"(16));
        }
    }
    __asm__ __volatile__("cp.async.commit_group;");

    __asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    call_cutlass_mma_prologue(T_matmul_cutlass_warp_mma, (&(p0_shared[0])), (&(p1_shared[0])), 32, 64);
    call_cutlass_mma_body(T_matmul_cutlass_warp_mma);
    for (int k_0 = 0; k_0 < 30; ++k_0)
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
                    "l"((void *)(p0 + (((((((((int)blockIdx.x) * 131072) + (ax0_ax1_fused_0_0_0_4 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16));
            }
        }
#pragma unroll
        for (int ax0_ax1_fused_0_0_0_5 = 0; ax0_ax1_fused_0_0_0_5 < 2; ++ax0_ax1_fused_0_0_0_5)
        {

            {
                unsigned int addr;
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(p1_shared + ((((((((k_0 & 1) * 2048) + (ax0_ax1_fused_0_0_0_5 * 1024)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.y) & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(p1 + (((((((((int)blockIdx.x) >> 3) * 65536) + (k_0 * 2048)) + (ax0_ax1_fused_0_0_0_5 * 1024)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 4096))), "n"(16));
            }
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 1;");

        __syncthreads();
        call_cutlass_mma_prologue(T_matmul_cutlass_warp_mma, (&(p0_shared[(((k_0 + 1) & 1) * 4096)])), (&(p1_shared[(((k_0 + 1) & 1) * 2048)])), 32, 64);
        call_cutlass_mma_epilogue(T_matmul_cutlass_warp_mma);
        call_cutlass_mma_body(T_matmul_cutlass_warp_mma);
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    call_cutlass_mma_prologue(T_matmul_cutlass_warp_mma, (&(p0_shared[4096])), (&(p1_shared[2048])), 32, 64);
    call_cutlass_mma_epilogue(T_matmul_cutlass_warp_mma);
    call_cutlass_mma_body(T_matmul_cutlass_warp_mma);
    call_cutlass_mma_epilogue(T_matmul_cutlass_warp_mma);
    __syncthreads();
#pragma unroll
    for (int ax1_0 = 0; ax1_0 < 32; ++ax1_0)
    {
        *(uint1 *)(T_matmul + (((((((((int)threadIdx.y) >> 1) * 4608) + ((ax1_0 & 7) * 576)) + ((((int)threadIdx.x) >> 2) * 72)) + ((((int)threadIdx.y) & 1) * 32)) + ((ax1_0 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1 *)(T_matmul_cutlass_warp_mma + (ax1_0 * 2));
    }
    __syncthreads();
}

__device__ void Group2_1_sum_3(half *__restrict__ p0, half *__restrict__ p0_red, char *shared)
{
    half p0_red_local[1];
    half *p0_shared = (half *)(shared + 0);
    p0_red_local[0] = __float2half_rn(0.000000e+00f);
    for (int k3_outer = 0; k3_outer < 16; ++k3_outer)
    {
        __syncthreads();
        *(uint4 *)(p0_shared + (((int)threadIdx.x) * 8)) = *(uint4 *)(p0 + ((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 2048)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 32768));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 3072)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 49152));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 4096)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 65536));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 5120)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 81920));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 6144)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 98304));
        *(uint4 *)(p0_shared + ((((int)threadIdx.x) * 8) + 7168)) = *(uint4 *)(p0 + (((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 3) * 1024)) + (k3_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 114688));
        __syncthreads();
        for (int k3_inner = 0; k3_inner < 64; ++k3_inner)
        {
            p0_red_local[0] = (p0_red_local[0] + p0_shared[((((int)threadIdx.x) * 64) + k3_inner)]);
        }
    }
    __syncthreads();
    p0_red[((int)threadIdx.x)] = p0_red_local[0];
    __syncthreads();
}

__device__ void Group2_2_divide_4(half *__restrict__ p0, half *__restrict__ p1, half *__restrict__ T_divide, char *shared)
{
    half *p0_shared = (half *)p0;
    half *p1_shared = (half *)p1;
    T_divide[(((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2))] = (p0_shared[(((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2))] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 16)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 16)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 32)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 32)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 48)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 48)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1024)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1152)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1040)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1168)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1056)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1184)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1072)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1200)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2048)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2304)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2064)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2320)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2080)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2336)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2096)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2352)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3072)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3456)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3088)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3472)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3104)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3488)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3120)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3504)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4096)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4608)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4112)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4624)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4128)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4640)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4144)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4656)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5120)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5760)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5136)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5776)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5152)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5792)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5168)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5808)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6144)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6912)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6160)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6928)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6176)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6944)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6192)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6960)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7168)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8064)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7184)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8080)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7200)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8096)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7216)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8112)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 17)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 17)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 33)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 33)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 49)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 49)] / p1_shared[(((int)threadIdx.x) >> 3)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1025)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1153)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1041)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1169)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1057)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1185)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 1073)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 1201)] / p1_shared[((((int)threadIdx.x) >> 3) + 16)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2049)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2305)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2065)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2321)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2081)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2337)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 2097)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 2353)] / p1_shared[((((int)threadIdx.x) >> 3) + 32)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3073)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3457)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3089)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3473)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3105)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3489)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 3121)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 3505)] / p1_shared[((((int)threadIdx.x) >> 3) + 48)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4097)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4609)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4113)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4625)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4129)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4641)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 4145)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 4657)] / p1_shared[((((int)threadIdx.x) >> 3) + 64)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5121)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5761)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5137)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5777)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5153)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5793)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 5169)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 5809)] / p1_shared[((((int)threadIdx.x) >> 3) + 80)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6145)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6913)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6161)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6929)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6177)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6945)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 6193)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 6961)] / p1_shared[((((int)threadIdx.x) >> 3) + 96)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7169)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8065)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7185)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8081)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7201)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8097)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
    T_divide[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + 7217)] = (p0_shared[((((((int)threadIdx.x) >> 3) * 72) + ((((int)threadIdx.x) & 7) * 2)) + 8113)] / p1_shared[((((int)threadIdx.x) >> 3) + 112)]);
}

__global__ void __launch_bounds__(128) Group2(half *input0, half *input1, half *input2, half *output0)
{
    // yeah I think they just go through SMEM
    __shared__ char shared[34816];
    Group2_0_welder_matmul_2(input0, input1, (half *)(shared + 0), shared + 0);
    Group2_1_sum_3(input2, (half *)(shared + 18432), shared + 18432);
    Group2_2_divide_4((half *)(shared + 0), (half *)(shared + 18432), output0, shared + 18688);
}