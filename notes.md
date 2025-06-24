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
