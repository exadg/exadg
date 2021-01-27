##
#  CMake script
##

PROJECT(poisson_gaussian)

EXADG_PICKUP_EXE(solver.cpp poisson_gaussian solver)

ADD_SUBDIRECTORY(tests)
