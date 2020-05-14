#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o horovod_o_output.o%j
#SBATCH -e horovod_e_output.e%j
#SBATCH -p gtx
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source ./maverick_init.sh


python3 gan_test.py
echo "Complete"
