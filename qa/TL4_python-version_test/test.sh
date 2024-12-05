#!/bin/bash -e
# used pip packages
export DALI_INSTALL_FROM_PIP=1

pip_packages='${python_test_runner_package} tensorflow-gpu'
target_dir=./dali/test/python

test_body() {
     #LATEST_DALI_VERSION="1.41.0"

     for $PACKAGE in "nvidia-dali", "nvidia-dali-tf-plugin"; do
         PIP_DALI_VERSION=`pip list | grep $(PACKAGE) | head -n 1 | awk '{print $2}'`

     	 if [[ "$LATEST_DALI_VERSION" != "$PIP_DALI_VERSION" ]]
             echo "Unexpected $(PACKAGE) version is: $(PIP_DALI_VERSION) shold be: $(LATEST_DALI_VERSION)"
	     exit 1
	 fi
     done

     exit 0
}

pushd ../..
source ./qa/test_template.sh
popd
