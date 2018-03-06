#!/bin/bash

process () {
    # arguments
    #   $1 nr of processes
    #   $2 dim
    #   $3 cells
    #   $4 points in cell
    #   $5 evalution points per cell
    python scripts/morton-test.py 0 $2 $3 $4 build/A
    python scripts/morton-test.py 1 $2 $3 $5 build/B
    mpirun -np $1 ./build/main -eval $5 build/A > build/temp
    python scripts/morton-diff.py build/A_converted build/B
}

#process 5 2 4 4 4
#process 5 2 4 4 5
#process 5 2 4 4 6
#
#process 5 2 8 4 4
#process 5 2 8 4 5
#process 5 2 8 4 6
#
#process 5 2 16 5 4
#process 5 2 16 5 5
#process 5 2 16 5 6
#
process 5 2 64 5 6

process 5 3 2 4 4
process 5 3 4 4 4
process 5 3 4 5 5
