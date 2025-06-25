__device__ void Group2_0_max_2(half* __restrict__ p0, half* __restrict__ p0_red, char* shared) {
  half normal_reduce_temp0[1];
  half* p0_shared = (half*)(shared+0);
  __shared__ half red_buf0[128];
  normal_reduce_temp0[0] = __float2half_rn(-6.550400e+04f);

  // copy p0 into p0_shared. Copies 8 elements per thread --> 1024 per block and note we just copy a row of p0
  *(uint4*)(p0_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(p0 + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)));
  __syncthreads();

  // reduces into temp0
  for (int k3_inner_outer = 0; k3_inner_outer < 8; ++k3_inner_outer) {
    normal_reduce_temp0[0] = max(normal_reduce_temp0[0], p0_shared[((k3_inner_outer * 128) + ((int)threadIdx.x))]);
  }
  __syncthreads();

  // we have a shared buffer of 128, so each thread saves to it
  ((volatile half*)red_buf0)[((int)threadIdx.x)] = normal_reduce_temp0[0];
  __syncthreads();

  // this is like a shuffle but through SMEM, so inefficient.
  if (((int)threadIdx.x) < 64) {
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 64)]));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 32)]));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    half w_16_0 = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 16)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_16_0;
    half w_8_0 = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 8)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_8_0;
    half w_4_0 = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 4)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_4_0;
    half w_2_0 = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 2)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_2_0;
    half w_1_0 = max((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]), (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 1)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_1_0;
  }
  __syncthreads();
  __syncthreads();
  p0_red[0] = (half)(((volatile half*)red_buf0)[0]);
  __syncthreads();
}

__device__ void Group2_1_subtract_exp_3(half* __restrict__ p0, half* __restrict__ p1, half* __restrict__ T_exp, char* shared) {
  half* p1_shared = (half*)p1;
  half p1_shared_local[1];
  p1_shared_local[0] = p1_shared[0];
  __syncthreads();
  T_exp[(((int)threadIdx.x) * 2)] = hexp((p0[((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2))] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 256)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 256)] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 512)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 512)] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 768)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 768)] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 1)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 1)] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 257)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 257)] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 513)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 513)] - p1_shared_local[0]));
  T_exp[((((int)threadIdx.x) * 2) + 769)] = hexp((p0[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 769)] - p1_shared_local[0]));
  __syncthreads();
}

__device__ void Group2_2_sum_4(half* __restrict__ p0, half* __restrict__ p0_red, char* shared) {
  half normal_reduce_temp0[1];
  half* p0_shared = (half*)p0;
  __shared__ half red_buf0[128];
  normal_reduce_temp0[0] = __float2half_rn(0.000000e+00f);
  for (int k3_inner_outer = 0; k3_inner_outer < 8; ++k3_inner_outer) {
    normal_reduce_temp0[0] = (normal_reduce_temp0[0] + p0_shared[((k3_inner_outer * 128) + ((int)threadIdx.x))]);
  }
  __syncthreads();
  ((volatile half*)red_buf0)[((int)threadIdx.x)] = normal_reduce_temp0[0];
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 64)]));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 32)]));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    half w_16_0 = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 16)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_16_0;
    half w_8_0 = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 8)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_8_0;
    half w_4_0 = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 4)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_4_0;
    half w_2_0 = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 2)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_2_0;
    half w_1_0 = ((half)(((volatile half*)red_buf0)[((int)threadIdx.x)]) + (half)(((volatile half*)red_buf0)[(((int)threadIdx.x) + 1)]));
    ((volatile half*)red_buf0)[((int)threadIdx.x)] = w_1_0;
  }
  __syncthreads();
  __syncthreads();
  p0_red[0] = (half)(((volatile half*)red_buf0)[0]);
  __syncthreads();
}

__device__ void Group2_3_divide_5(half* __restrict__ p0, half* __restrict__ p1, half* __restrict__ T_divide, char* shared) {
  half* p0_shared = (half*)p0;
  half* p1_shared = (half*)p1;
  T_divide[((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2))] = (p0_shared[(((int)threadIdx.x) * 2)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 256)] = (p0_shared[((((int)threadIdx.x) * 2) + 256)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 512)] = (p0_shared[((((int)threadIdx.x) * 2) + 512)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 768)] = (p0_shared[((((int)threadIdx.x) * 2) + 768)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 1)] = (p0_shared[((((int)threadIdx.x) * 2) + 1)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 257)] = (p0_shared[((((int)threadIdx.x) * 2) + 257)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 513)] = (p0_shared[((((int)threadIdx.x) * 2) + 513)] / p1_shared[0]);
  T_divide[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 2)) + 769)] = (p0_shared[((((int)threadIdx.x) * 2) + 769)] / p1_shared[0]);
}

__global__ void __launch_bounds__(128) Group2(half* input0, half* input1, half* output0) {
  __shared__ char shared[2080];
  Group2_0_max_2(input1, (half*)(shared+0), shared+0);
  Group2_1_subtract_exp_3(input0, (half*)(shared+0), (half*)(shared+0), shared+32);
  Group2_2_sum_4((half*)(shared+0), (half*)(shared+2048), shared+2048);
  Group2_3_divide_5((half*)(shared+0), (half*)(shared+2048), output0, shared+2080);
}