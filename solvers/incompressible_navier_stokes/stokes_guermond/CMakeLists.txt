##
#  CMake script
##

PROJECT(incompressible_stokes_guermond)

EXADG_PICKUP_EXE(solver.cpp incompressible_guermond solver)
