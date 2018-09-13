#!/bin/bash

pwd=$(pwd)

make_test () {
  cd $1
  mkdir -p build
  cd build
  pwd
  cmake ..
  make -j15
  cd $pwd
}

make_test operators/operation-base-1
make_test operators/operation-base-2
make_test operators/operation-base-3
make_test operators/operation-base-4

make_test transfer/dg-to-cg-transfer-1/
make_test transfer/dg-to-cg-transfer-2/
make_test transfer/dg-to-cg-transfer-3/

make_test transfer/p-transfer-1/
make_test transfer/p-transfer-2/
