#!/bin/bash
array=($(ls output/filename_energy_spectrum_0*)) 
mpirun -np xxx ./deal-spectrum "${array[@]}"

