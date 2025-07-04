- when compiling TVM use -j 1 to not run out of memory if you get these process terminated errors
- so for `run_compiler` you need json
- python3 -m pip install regex
- currently trying to figure out their relay test
- in welder/arch/cuda.py, set compute capability to 86 by default. They weren't ready to accomodate arch 120
- ok so it runs into errors lowering a tanh or something later on, but I think I have what I need. Next step is to get it up and running with just GEMM or something to test tiling, then with flash attention to see how that lowers

# Overall observations
- when we add transposes to the graph, it treats it like a node I think like it actually does the transpose before feeding into like matmul or smth and creates fusion groups like [transpose_0, transpose_welder_matmul_divide_1]

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

### Codebase - Attention Example
TODO look for why the policies didn't emit anything for transpose_0 + matmul_divide_1.

- First node is a transpose, so the first candidate group is `[<Node, transpose_0>, <Node, welder_matmul_divide_1>]`
    - Notice how the matmul was already fused to divide by a previous pass
- **TCPolicy emits nothing cuz it's a transpose, seems like default policy emits nothing too. Hmm...**
    - I think it's because they treat the transpose as a whole operation but we gotta look into it more
- Transpose policy is e.g. `{'globals': {'Rasterization': <NoRasterization>}, <Node, transpose_0>: {'block': [2, 1, 64, 64], 'thread': [2, 1, 8, 8], 'rstep': [], 'step': [1, 1, 1, 2]}}`
    - so remember since the tensor is 4D and we're transposing the last 2 dims, and there's no rsteps
- then it tries tuning matmul_divide_1 + max + subtract but it then tunes matmul_divide_1
- NEED to look more at tuning policy etc. plus how it selects what to tune

### How does their higher level build_fusion_group work?
- iterates through ordered nodes, so starts with tranpose_0
- ok I get it, the reason we were seeing initial fusion groups that are large is because of the branching. It tries to fuse with ALL outputs on a first step.

### How does their lower-level fusion stuff work
- they try to fuse tranpose_0 to matmul_divide_1, but it never generates SMEM_tile_candidates.
- What I'm getting is they have an initial tile that they try to expand on. Their initial SMEM tile for transpose_0 + matmul_divide_1 is too big so they just don't do it.
- so since they just don't model the transpose that well, it doesn't fuse with matmul. I can think about that but doesn't seem too innovative

### Dissecting tuned softmax
['max_2', 'subtract_exp_3', 'sum_4', 'divide_5']

`{'globals': {'Rasterization': <NoRasterization>}, <Node, max_2>: {'block': [1, 1, 1, 1], 'thread': [1, 1, 1, 1], 'rstep': [1024], 'reduce_thread': [128], 'vectorize': {'p0': 8}}, <Node, subtract_exp_3>: {'block': [1, 1, 1, 1024], 'thread': [1, 1, 1, 128], 'rstep': [], 'step': [1, 1, 1, 2]}, <Node, sum_4>: {'block': [1, 1, 1, 1], 'thread': [1, 1, 1, 1], 'rstep': [1024], 'reduce_thread': [128], 'vectorize': {'p0': 8}}, <Node, divide_5>: {'block': [1, 1, 1, 1024], 'thread': [1, 1, 1, 128], 'rstep': [], 'step': [1, 1, 1, 2]}}`

block size 128, grid size 262144 = (16 * 16 * 1024). This is for the ENTIRE fused kernel btw

- So each thread reduces 8 elements in a split-k scheme, such that 128 threads reduce 1024 elements. Even when they only tune max they have this kinda setup. I wonder how good that is, since I haven't done a max kernel before
- then subtract_exp, no rstep and it seems like each thread does 128 elements, each block does 1024(???) that doesn't sound right but ok
- Then the sum where they reduce along each axis is also just 128 threads to reduce so each thread reduces 8 elements
- then the divide, seems like each block takes 1024 elements, each thread takes 128(??) and yeah.

So overall each block takes a single row of the matrix. That does make sense since it's not like larger tiles give more opportunity for reuse

### Why does Welder fail on "Easy Attention?"
- Fails to generate configs, look at welder/policy/default.py
- `[<Node, welder_matmul_divide_exp_1>, <Node, welder_matmul_2>, <Node, sum_3>]` are the node's it tries to fuse first
- It does generate a base tile [1, 1, 1, 64]
- `{<Node, welder_matmul_divide_exp_1>: {'k': 32}, <Node, welder_matmul_2>: {'k': 32}, <Node, sum_3>: {'k3': 1}}` is the rstep generated
- it generates no smem tile candidates from DFS_SMEM_TILE

DFS_SMEM_TILE
- first start with _steps which is all factors, but narrow down based on min tile size
- `[[1, 2, 4, 8, 16], [1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [64]]` since the min tile size is 64 for the last dim(I wonder how they got that but okok)
- So, the first tile dict was invalid
- let's look at compute_tile_dict for that tile(from add_to_queue in DFS_SMEM_TILE)
- So it initializes a tile dict with output_shape as the tile shape we put in (64 tile)
- Wait this is literally what FlashAttention has though, except I don't know what that k3 sum reduction dim is and why it's 1
- tile map: `{<Node, welder_matmul_2>: [1, 1, 1, 64], <Node, sum_3>: [1, 1, 1, 1], <Node, welder_matmul_divide_exp_1>: [1, 1, 1, 1024], <Node, PlaceHolder >: [1, 1, 1024, 64], <Node, PlaceHolder >: [1, 1, 1, 64], <Node, PlaceHolder >: [1, 1, 64, 1024]}`
    - So first matmul(matmul_divide_exp) it's gonna output a 1x1024 tile???
    - Then into the matmul 2 we're gonna output a 1x64 tile so 1x1024x1024x64 ahh I see that's not good...
    - And then the sum we're gonna output a 1x1 tile but reduce down the 1024 dim I'm guessing
    - **ALSO** the thing that actually returns the tile map is `_compute_memory_traffic()`, I think their propagation logic there is flawed or smth
    - the placeholders are the outputs(??) I think?? I think the 64x1024 is the size of the P matrix from the first matmul, 1024x64 is size of O matrix after second matmul and 1, 64 is the rowsum???

How does it get the tile map?
- initial output tile map `{<Node, welder_matmul_2>: [1, 1, 1, 64], <Node, sum_3>: [1, 1, 1, 1]}` these are min tile sizes
- so when they propagate the sum shape back, it becomes [1, 1, 1, 1024]
- The matmul_2 expects input shapes `[[1, 1, 1, 1024], [1, 1, 1024, 64]]` like I think they should have a reduction step but idk anymore...
- so when it's calculating traffic they're literally doing `+= [1024 * 64 * nbytes]` but idk what traffic cost means so...
- for the first matmul divide exp, they have these input shapes: `[[1, 1, 1, 64], [1, 1, 64, 1024]]`
- they calculate `smem_cost=66144` so it's too big for smem I think
- ok YEAH based on `infer_node_smem_usage` it seems like the tile is literally the tile, and rstep_map is rstep_map so they're literally trying to fit like a 1024x64 because they can't splitK on the second matmul since it's preceded by the first matmul
- so then there's too much smem usage and we're done, no configs generated.
- I feel like if they had a higher level representation that could be good, but there's so much code to express something that's simple imo. Like they're doing so much optimization from one representation like calculating smem usage etc. Like their propagate and stuff is interleaved inside this entire algorithm so you can't just modify it that easily, it's hard to mess with their graph structure. The code design is hard to extend e.g. if I wanted to add more stuff to the representation.

### Will manually reorganizing the graph lead to good fusion?
- `[<Node, transpose_0>, <Node, welder_matmul_divide_exp_1>, <Node, sum_3>, <Node, welder_matmul_2>, <Node, divide_4>, <Node, Output >]`
- Reordered the sum and matmul2, which occur due to branching so you can reorder them in the toposort
- 