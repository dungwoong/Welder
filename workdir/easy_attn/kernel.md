### welder matmul divide exp 1
```
{
    'globals': {'Rasterization': <NoRasterization>}, 
    <Node, welder_matmul_divide_exp_1>: 
        {'block': [1, 2, 64, 64], 
        'warp': [1, 2, 32, 32], 
        'wmma': [16, 16, 16], 
        'use_cutlass': False, 
        'rstep': [32], 
        'use_tc': '86', 
        'strides': {2: <Stride, 2, 72>
}}}
```

- Each block does 64x64 but for 2 heads actually
- each warp does 32x32
- reduce of k dim of 32
- This should be a decently performant GEMM kernel, seems fine...

### matmul 2 + sum 3 + divide 4
```
{'globals': {'Rasterization': <NoRasterization>}, 
<Node, welder_matmul_2>: {'block': [1, 1, 128, 64], 'warp': [1, 1, 64, 32], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '86', 'strides': {2: <Stride, 2, 72>}}, 
<Node, sum_3>: {'block': [1, 1, 128, 1], 'thread': [1, 1, 128, 1], 'rstep': [64], 'vectorize': {'p0': 8}}, 
<Node, divide_4>: {'block': [1, 1, 128, 64], 'thread': [1, 1, 16, 8], 'rstep': [], 'step': [1, 1, 1, 2]}}
```

- The matmul 2 uses cutlass and each block/warp does (128, 64)/(64, 32) and reduces over 32 elements
- the sum reduces (128, 64) per block --> (128, 1) per thread
- the divide  (128, 64) per block --> (16, 8) per thread
- ok so I don't think they're directly feeding into each other here, might be going through SMEM
- looking at the CUDA code, it seems like it's going through SMEM since it has a global kernel that only declares `__shared__` array, and it just calls 3 device functions for each thing

Here is what the kernel looks like:

```c++
__global__ void __launch_bounds__(128) Group2(half* input0, half* input1, half* input2, half* output0) {
  __shared__ char shared[34816];
  Group2_0_welder_matmul_2(input0, input1, (half*)(shared+0), shared+0);
  Group2_1_sum_3(input2, (half*)(shared+18432), shared+18432);
  Group2_2_divide_4((half*)(shared+0), (half*)(shared+18432), output0, shared+18688);
}
```

- so it's not even doing things inside one inner loop. It's literally executing the kernels serially, except using SMEM instead of GMEM to pass things.
- why do I feel like they never actually fuse at a register level and they leave that to other compiler passes(eg. how the first matmul gets fused to divide and exp automatically...)