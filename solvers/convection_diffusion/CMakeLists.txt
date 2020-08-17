##
#  CMake script
##

PROJECT(convection_diffusion)

# string replace: cut off .cc from files
ADD_EXECUTABLE( convection_diffusion solver.cpp)
DEAL_II_SETUP_TARGET(convection_diffusion)
set_target_properties(convection_diffusion PROPERTIES OUTPUT_NAME "solver")
TARGET_LINK_LIBRARIES(convection_diffusion exadg)

ADD_EXECUTABLE( convection_diffusion_throughput throughput.cpp)
DEAL_II_SETUP_TARGET(convection_diffusion_throughput)
set_target_properties(convection_diffusion_throughput PROPERTIES OUTPUT_NAME "throughput")
TARGET_LINK_LIBRARIES(convection_diffusion_throughput exadg)
