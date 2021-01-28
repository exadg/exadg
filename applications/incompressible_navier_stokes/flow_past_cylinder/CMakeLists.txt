##
#  CMake script
##

PROJECT(incompressible_flow_past_cylinder)

EXADG_PICKUP_EXE(solver.cpp incompressible_flow_past_cylinder solver)
