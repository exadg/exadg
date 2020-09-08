##
#  CMake script
##

PROJECT(incompressible_navier_stokes)

EXADG_PICKUP_EXE(solver.cpp incompressible_navier_stokes solver)
EXADG_PICKUP_EXE(solver_precursor.cpp incompressible_navier_stokes_precursor solver_precursor)
EXADG_PICKUP_EXE(throughput.cpp incompressible_navier_stokes_throughput throughput)
