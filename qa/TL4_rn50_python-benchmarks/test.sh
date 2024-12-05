#!/bin/bash -e
# used pip packages
pip_packages='numpy'
target_dir=./dali/benchmark
export DALI_INSTALL_FROM_PIP=1

test_body() {
    # test code
    python resnet50_bench.py
}

pushd ../..
source ./qa/test_template.sh
popd
