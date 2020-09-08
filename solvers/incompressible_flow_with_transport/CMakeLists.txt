##
#  CMake script
##

PROJECT(incompressible_flow_with_transport)

EXADG_PICKUP_EXE(solver.cpp incompressible_flow_with_transport solver)
