#!/bin/bash

mkdir -p tvm/build && cd tvm/build && cp ../cmake/config.cmake . && cmake .. && make -j 4 && cd -

export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PWD/cutlass/include"
export PYTHONPATH="$PYTHONPATH:$PWD/tvm/python"
export PYTHONPATH="$PYTHONPATH:$PWD/python"

python3 ./testing/run_all_experiments.py