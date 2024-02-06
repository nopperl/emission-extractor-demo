#!/bin/sh
git clone https://github.com/nopperl/llama.cpp
if [ ! -f llama.cpp/build/bin/main ]; then
  cd llama.cpp
  mkdir build
  cd build
  cmake ..
  cmake --build . --config Release
fi
