#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o output_o.o%j
#SBATCH -e output_e.e%j
#SBATCH -p p100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 20:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH --mail-type=all
#SBATCH -A COSC4381Spring2020


echo "Setting env for Maverick2"
source ./init_maverick.sh


python3 gan_test.py
echo "Complete"
