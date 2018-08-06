#!/bin/bash

#SBATCH -o /home/hpc/t7341/ga69jux3/swwww/nav-ref/tests/run-scripts/job.out
#SBATCH -D /home/hpc/t7341/ga69jux3/swwww/nav-ref/tests/run-scripts

#SBATCH -J LIKWID
#SBATCH --get-user-env
#SBATCH --clusters=mpp2
#SBATCH --ntasks=28
#SBATCH --mail-type=end
#SBATCH --mail-user=ga69jux@mytum.de
#SBATCH --export=NONE
#SBATCH --time=00:20:00

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

# mpirun -np 28 ../transfer/dg-to-cg-transfer-3/build/main | tee out-cg-dg.txt
# mpirun -np 28 ../transfer/p-transfer-2/build/main        | tee out-p.txt


#/home/hpc/t7341/ga69jux3/projects/likwid-mpirun/scripts/likwid-mpirun -O -f -np 28 -g CACHES    -m ../transfer/dg-to-cg-transfer-3/build/main | tee out-cg-dg.txt
 /home/hpc/t7341/ga69jux3/projects/likwid-mpirun/scripts/likwid-mpirun -O -f -np 28 -g CACHES    -m ../transfer/p-transfer-2/build/main        | tee out-p.txt

