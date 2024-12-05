#!/bin/bash -e
DALI_INSTALL_FROM_PIP=1
bash -e ../TL0_cpu_only/test_pytorch.sh
bash -e ../TL0_cpu_only/test_tf.sh
bash -e ../TL0_cpu_only/test_nofw.sh
