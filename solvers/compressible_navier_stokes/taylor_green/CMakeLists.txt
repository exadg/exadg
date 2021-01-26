##
#  CMake script
##

PROJECT(compressible_taylor_green)

EXADG_PICKUP_EXE(solver.cpp compressible_taylor_green solver)
EXADG_PICKUP_EXE(throughput.cpp compressible_throughput throughput)