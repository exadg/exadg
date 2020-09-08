##
#  CMake script
##

PROJECT(compressible_navier_stokes)

EXADG_PICKUP_EXE(solver.cpp compressible_navier_stokes solver)
EXADG_PICKUP_EXE(throughput.cpp compressible_navier_stokes_throughput throughput)
