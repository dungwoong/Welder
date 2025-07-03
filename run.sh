#!/bin/bash

export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PWD/cutlass/include"
export PYTHONPATH="$PYTHONPATH:$PWD/tvm/python"
export PYTHONPATH="$PYTHONPATH:$PWD/python"

python3.12 ./testing/run_all_experiments.py