##
#  CMake script
##

PROJECT(incompressible_flow_with_transport)

# string replace: cut off .cc from files
ADD_EXECUTABLE( incompressible_flow_with_transport solver.cpp)
DEAL_II_SETUP_TARGET(incompressible_flow_with_transport)
set_target_properties(incompressible_flow_with_transport PROPERTIES OUTPUT_NAME "solver")
TARGET_LINK_LIBRARIES(incompressible_flow_with_transport exadg)
