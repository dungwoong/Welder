- when compiling TVM use -j 1 to not run out of memory if you get these process terminated errors
- so for `run_compiler` you need json
- python3 -m pip install regex
- currently trying to figure out their relay test
- in welder/arch/cuda.py, set compute capability to 86 by default. They weren't ready to accomodate arch 120
- ok so it runs into errors lowering a tanh or something later on, but I think I have what I need. Next step is to get it up and running with just GEMM or something to test tiling, then with flash attention to see how that lowers

# Figuring it out

### Torch2onnx on random model like gemm or attention
- edit `testing/model/pytorch/__init__.py` and add anything we need

### Run welder on it
- python3 ./testing/relay_test.py workdir and it'll run on your onnx model
- different than what victor had, there was na artifacts folder and stuff and idk how that works

### Codebase - Matmul Example
- in `welder/engine/base_tunner`, extract_subgraph will return input and output desc. These are just lists that enumerate the input nodes I think like `[['nn_matmul_0', 0], ['nn_matmul_0', 1]]`
- `input_desc=[['nn_matmul_0', 0], ['nn_matmul_0', 1]] output_desc=[['nn_matmul_0', 0]]` so this is the output of extract subgraph
- `policy_list=[<class 'welder.policy.tc.TCPolicy'>, <class 'welder.policy.default.DefaultPolicy'>]`
- It generates 20 configs(TODO check out `generate_configs`)

Checking out a sample config
- has keys `globals` and `<Node, nn_matmul_0>` (so for more nodes, we'd probably have more configs)
- global is eg. `configs[0]['globals']={'Rasterization': <Rasterization2DRow(12)>}` with column size and row size 32
- then linked to the node, we have this info: `{'block': [128, 128], 'warp': [64, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '86', 'strides': {2: <Stride, 0, 136>}}`
- so each block does 128 x 128, each warp 64x64, wmma is 16x8x16. Rstep is probably reduction step so it reduces 32 everytime. Use TC is just use tensor core, idk about strides
- These are `Config` objects. Also you can do compile_results.index(best) and then look at that specific config for the matching config I think.
- ok so the best result is does correspond to the config we have above.

- You can look at gemm4096_with_ref.cu. So yeah what I had in mind was basically what they have for GEMM. We can mess with the cost model to make tuning a bit faster maybe but we'll see.
- look at `welder/policy/default.py`. That's where they do the smem candidates and then add tiling for warps and stuff.

### Overall observations
- when we add transposes to the graph, it treats it like a node I think like it actually does the transpose before feeding into like matmul or smth and creates fusion groups like [transpose_0, transpose_welder_matmul_divide_1]