##
#  CMake script
##

PROJECT(structure)

EXADG_PICKUP_EXE(solver.cpp structure solver)
EXADG_PICKUP_EXE(throughput.cpp structure_throughput throughput)
