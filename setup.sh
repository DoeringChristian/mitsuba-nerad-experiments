#!/bin/bash

CLANG=~/programms/clang14/bin/clang
CLANGXX=~/programms/clang14/bin/clang++

source ./.venv/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
# export CPATH="/usr/local/cuda-11.8/include"
export CC=$CLANG
export CXX=$CLANGXX

pip install -r requirements.txt

