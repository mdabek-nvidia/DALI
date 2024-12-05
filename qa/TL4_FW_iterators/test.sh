#!/bin/bash -e
export DALI_INSTALL_FROM_PIP=1
bash -e ../TL0_FW_iterators/test_tf.sh
bash -e ../TL0_FW_iterators/test_paddle.sh
bash -e ../TL0_FW_iterators/test_pytorch.sh
bash -e ../TL0_FW_iterators/test_jax.sh
