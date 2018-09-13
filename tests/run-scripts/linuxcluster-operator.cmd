#!/bin/bash

#SBATCH -o ./linuxcluster-operator.%j.%N.out 
#SBATCH -D .

#SBATCH -J LIKWID
#SBATCH --get-user-env
#SBATCH --clusters=mpp2
#SBATCH --ntasks=28
#SBATCH --mail-type=end
#SBATCH --mail-user=ga69jux@mytum.de
#SBATCH --export=NONE
#SBATCH --time=01:00:00

# LOAD MODULES
#source /etc/profile.d/modules.sh
module unload mpi.intel
module unload intel/17.0
module load git
module load intel/17.0
module load mpi.intel/2017
module load gcc/7
module load likwid/4.1

module list

# RUN PROGRMAMM

mkdir -p results

mpirun -np 28 ../operators/operation-base-3/build/main | tee results/out-operation.txt
