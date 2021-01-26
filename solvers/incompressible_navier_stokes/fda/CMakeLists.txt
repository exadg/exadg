##
#  CMake script
##

PROJECT(incompressible_fda)

EXADG_PICKUP_EXE(solver.cpp incompressible_fda solver)
EXADG_PICKUP_EXE(solver_precursor.cpp incompressible_fda_precursor solver_precursor)