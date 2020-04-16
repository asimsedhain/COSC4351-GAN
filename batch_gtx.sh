#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o output_o.o%j
#SBATCH -e output_e.e%j
#SBATCH -p gtx
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source ./gtx_maverick_init.sh

export IBRUN_TASKS_PER_NODE=4

ibrun -np 8 python3 gan_test.py
echo "Complete"
