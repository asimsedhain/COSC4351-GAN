#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o output_o.o%j
#SBATCH -e output_e.e%j
#SBATCH -p gtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020


echo "Setting env for Maverick2"

module load intel/17.0.4 python3/3.6.3

echo "Python Loaded"

module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7

echo "Cuda Loaded"

pip3 install --user tensorflow-gpu==1.13.2

echo "Tensorflow Loaded"


pip3 install --user keras


echo "Keras Loaded"

pip3 install --user h5py

echo "h5py Loaded"

CPLUS_INCLUDE_PATH=/opt/apps/intel18/impi18_0/boost/1.66/include \
                CC=gcc HOROVOD_CUDA_HOME=/opt/apps/cuda/10.0 HOROVOD_GPU_ALLREDUCE=NCCL \
                HOROVOD_NCCL_HOME=/opt/apps/cuda10_0/nccl/2.4.7 HOROVOD_WITH_TENSORFLOW=1 \
                HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip3 install \
                --user horovod==0.16.4 --no-cache-dir
echo "env varibales Loaded"

pip install matplotlib==2.0.0

echo "safe matplotlib version loaded"

python3 gan_test.py
echo "Complete"
