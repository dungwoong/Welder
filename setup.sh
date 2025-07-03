#!/bin/bash

# assume TVM and cutlass paths are ready
mkdir -p tvm/build && cd tvm/build && cp ../cmake/config.cmake . && cmake .. && make -j && cd -