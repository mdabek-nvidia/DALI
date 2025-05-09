#!/bin/bash -e

# used pip packages
pip_packages='jupyter matplotlib numpy'
target_dir=./docs/examples/

test_body() {
    test_files=("sequence_processing/optical_flow_example.ipynb")

    # test code
    echo $test_files | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=600 {}
}

pushd ../..
source ./qa/test_template.sh
popd
