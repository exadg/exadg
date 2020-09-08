##
#  CMake script
##

PROJECT(poisson)

EXADG_PICKUP_EXE(solver.cpp poisson solver)
EXADG_PICKUP_EXE(throughput.cpp poisson_throughput throughput)

ADD_SUBDIRECTORY(tests)
