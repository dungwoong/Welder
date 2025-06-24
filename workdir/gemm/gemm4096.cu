// blocksize 32, 4, 1
// gridsize 512

// seems like each warp does 64x128, so then *4 warps * 512 blocks = 4096 * 4096.
// - A(64 * 128) * B(128 * 128)
// - 

__global__ void __launch_bounds__(128) Group0(half* __restrict__ p0, half* __restrict__ p1, half* __restrict__ T_matmul_NN) {
  int __bid = blockIdx.x;
  const dim3 blockIdx(rasterization2DColumn<32, 16, 8>(__bid), 0, 0);
  // m64n128k128
  // break down into 4x m64n128k32
    __shared__ half p0_shared[8192];
  __shared__ half p1_shared[16384];

  // m64n128k32 gemm tensor op
  ALLOCATE_CUTLASS_OBJECT(T_matmul_NN_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>
>((((int)threadIdx.y) >> 1), (((int)threadIdx.y) & 1), ((int)threadIdx.x)));


// this is fetching stage 1 or whatever
// notice that 4 fetches of 16B(8 elements) per thread * 128 threads = 4096 = (64 * 64)
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0 = 0; ax0_ax1_fused_0_0_0 < 4; ++ax0_ax1_fused_0_0_0) {

  {
    unsigned int addr;
    // cvta_generic_to_shared, referring to some location in p0_shared
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(p0_shared + (((((ax0_ax1_fused_0_0_0 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );

    // cp async shared global, 16B(128 bits)
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p0 + ((((((((int)blockIdx.x) >> 4) * 524288) + (ax0_ax1_fused_0_0_0 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }

  // this is fetching stage 1 for B. They fetch 64x128 for B?
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 8; ++ax0_ax1_fused_0_0_0_1) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(p1_shared + ((((((ax0_ax1_fused_0_0_0_1 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ax0_ax1_fused_0_0_0_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.y) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + (((int)threadIdx.y) & 1)) & 1) * 8))))
    );
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p1 + ((((ax0_ax1_fused_0_0_0_1 * 16384) + (((int)threadIdx.y) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 8)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_2 = 0; ax0_ax1_fused_0_0_0_2 < 4; ++ax0_ax1_fused_0_0_0_2) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(p0_shared + ((((((ax0_ax1_fused_0_0_0_2 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)))
    );
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p0 + (((((((((int)blockIdx.x) >> 4) * 524288) + (ax0_ax1_fused_0_0_0_2 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
  }
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_3 = 0; ax0_ax1_fused_0_0_0_3 < 8; ++ax0_ax1_fused_0_0_0_3) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(p1_shared + (((((((ax0_ax1_fused_0_0_0_3 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ax0_ax1_fused_0_0_0_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.y) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + (((int)threadIdx.y) & 1)) & 1) * 8)) + 8192)))
    );
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p1 + (((((ax0_ax1_fused_0_0_0_3 * 16384) + (((int)threadIdx.y) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 8)) + 131072))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  call_cutlass_mma_prologue(T_matmul_NN_cutlass_warp_mma, (&(p0_shared[0])), (&(p1_shared[0])), 32, 256);
  call_cutlass_mma_body(T_matmul_NN_cutlass_warp_mma);

  // THE MAIN LOOP, loops 126 times
  for (int k_0 = 0; k_0 < 126; ++k_0) {
    // fetch more stuff
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_4 = 0; ax0_ax1_fused_0_0_0_4 < 4; ++ax0_ax1_fused_0_0_0_4) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(p0_shared + (((((((k_0 & 1) * 4096) + (ax0_ax1_fused_0_0_0_4 * 1024)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p0 + ((((((((((int)blockIdx.x) >> 4) * 524288) + (ax0_ax1_fused_0_0_0_4 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
    }
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_5 = 0; ax0_ax1_fused_0_0_0_5 < 8; ++ax0_ax1_fused_0_0_0_5) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(p1_shared + ((((((((k_0 & 1) * 8192) + (ax0_ax1_fused_0_0_0_5 * 1024)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ax0_ax1_fused_0_0_0_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.y) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + (((int)threadIdx.y) & 1)) & 1) * 8))))
    );
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(p1 + ((((((k_0 * 131072) + (ax0_ax1_fused_0_0_0_5 * 16384)) + (((int)threadIdx.y) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 8)) + 262144))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    call_cutlass_mma_prologue(T_matmul_NN_cutlass_warp_mma, (&(p0_shared[(((k_0 + 1) & 1) * 4096)])), (&(p1_shared[(((k_0 + 1) & 1) * 8192)])), 32, 256);
    call_cutlass_mma_epilogue(T_matmul_NN_cutlass_warp_mma);
    call_cutlass_mma_body(T_matmul_NN_cutlass_warp_mma);
  }


  // END OF THE MAIN LOOP, do one more mma
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  call_cutlass_mma_prologue(T_matmul_NN_cutlass_warp_mma, (&(p0_shared[4096])), (&(p1_shared[8192])), 32, 256);
  call_cutlass_mma_epilogue(T_matmul_NN_cutlass_warp_mma);
  call_cutlass_mma_body(T_matmul_NN_cutlass_warp_mma);
  call_cutlass_mma_epilogue(T_matmul_NN_cutlass_warp_mma);

  // seems like each thread saves 8 items, so 1024 per block?
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 128; ++ax1_0) {
    *(uint1*)(T_matmul_NN + (((((((((((int)blockIdx.x) >> 4) * 524288) + ((((int)threadIdx.y) >> 1) * 262144)) + ((ax1_0 & 7) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.y) & 1) * 128)) + ((ax1_0 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(T_matmul_NN_cutlass_warp_mma + (ax1_0 * 2));
  }
}