##
#  CMake script
##

PROJECT(convection_diffusion)

EXADG_PICKUP_EXE(solver.cpp convection_diffusion solver)
EXADG_PICKUP_EXE(throughput.cpp convection_diffusion_throughput throughput)
