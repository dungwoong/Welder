#!/bin/bash

export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PWD/cutlass/include"
export PYTHONPATH="$PYTHONPATH:$PWD/tvm/python"
export PYTHONPATH="$PYTHONPATH:$PWD/python"

python3 ./testing/torch2onnx.py selfattneasy --bs 16 --fp16 --prefix workdir
python3 ./testing/relay_test.py workdir
python3 ./testing/test_welder_perf.py workdir
python3 ./testing/test_welder_acc.py workdir
python3 ./testing/kevin_test_perf.py workdir