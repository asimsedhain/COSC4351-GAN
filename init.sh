#!/bin/bash

module load intel/18.0.2 python3/3.7.0

pip3 install --user /home1/apps/tensorflow/builds/intel-18.0.2/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.whl

pip3 install --user keras

pip3 install --user --force-reinstall h5py --no-deps

export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages:/opt/apps/intel18/impi18_0/python3/3.7.0/lib/python3.7/site-packages

module load boost/1.68

CPLUS_INCLUDE_PATH=/opt/apps/intel18/boost/1.68/include HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip3 install --user horovod==0.16.4 --no-cache-dir

