#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J filename.16-node
#Output and error (also --output, --error):
#SBATCH -o ./path/to/output/directory/%x.%j.out
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=<email-address>
# Wall clock limit:
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --account=<project-ID>
#SBATCH --get-user-env
#SBATCH --ear=off
## test, micro, general, large
#SBATCH --partition=test
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=48

source ~/.bashrc

lscpu

mpiexec.hydra ./build/path/to/exe ./path/to/input.json
