##
#  CMake script
##

PROJECT(fda_benchmark)

EXADG_PICKUP_EXE(solver_poisson.cpp poisson_fda solver_poisson)
EXADG_PICKUP_EXE(solver.cpp incompressible_fda solver)
EXADG_PICKUP_EXE(solver_precursor.cpp incompressible_precursor_fda solver_precursor)