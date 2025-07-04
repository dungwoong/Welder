# DO NOT ACTUALLY RUN THIS, this is just notes for setting up
# clone
apptainer build --sandbox --fakeroot cuda128.sandbox docker://nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# get in apptainer
apptainer shell --nv --writable --fakeroot cuda128.sandbox/

# upgrade g++
apt update
apt install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt update
apt install gcc-11 g++-11

# set defaults to v11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
update-alternatives --config g++
update-alternatives --config gcc

# GET CMAKE
# download cmake-3.22.1-linux-x86_64.sh from https://github.com/Kitware/CMake/releases/tag/v3.22.1
# run it just on your host machine it should be fine
# put it into /opt
# ADD THIS TO .environment:
# export PATH=/opt/cmake-3.22.1-linux-x86_64/bin:$PATH
cmake --version # should be 3.22.1

# fa_gemm_minimal should work by now.

# Python3.12
# follow this https://www.debugpoint.com/install-python-3-12-ubuntu/
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.12
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
update-alternatives --config python3

# setup pip
# https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
# now pip3 = python3 -m pip

# Setting up welder stuff
#python
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip3 install onnx attrs cloudpickle decorator psutil synr tornado xgboost timm onnxruntime-gpu
pip3 install setuptools regex

# LLVM
apt update
apt-get install llvm-dev


# TRY RUN WELDER
# exit apptainer
git clone https://github.com/dungwoong/Welder.git
cd Welder
git clone https://github.com/nox-410/tvm --recursive -b develop
# set use_llvm and use_cuda in tvm/cmake/config.cmake
git clone https://github.com/nox-410/cutlass -b welder
# in tvm/python/tvm/_ffi/runtime_ctypes.py, delete the line with np.float_ (deprecated)

# build TVM
apptainer shell --nv --bind $PWD/Welder:/home/Welder cuda128.sandbox
cd /home/welder
mkdir -p tvm/build && cd tvm/build && cp ../cmake/config.cmake . && cmake .. && make -j 4 && cd -