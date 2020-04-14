#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o output_o.o%j
#SBATCH -e output_e.e%j
#SBATCH -p gtx
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020


echo "Setting env for Maverick2"
source ./gtx_maverick_init.sh


ibrun -np 16 python3 gan_test.py
echo "Complete"
