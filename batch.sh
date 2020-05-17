#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o ./output/singularity_attention/output_o.%j
#SBATCH -e ./output/singularity_attention/output_e.%j
#SBATCH -p gtx
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source ./maverick_init.sh


singularity exec --nv ../singularity_test/tacc-maverick_1.2.sif python $WORK/COSC4381/gan_test.py
echo "Complete"
