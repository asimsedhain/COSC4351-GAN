#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o horovod_p100_o_output.o%j
#SBATCH -e horovod_p100_e_output.e%j
#SBATCH -p p100
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source ./maverick_init.sh

export IBRUN_TASKS_PER_NODE=2

ibrun -np 2 python3 gan_test.py
echo "Complete"
