##
#  CMake script
##

PROJECT(structure)

# string replace: cut off .cc from files
ADD_EXECUTABLE( structure solver.cpp)
DEAL_II_SETUP_TARGET(structure)
set_target_properties(structure PROPERTIES OUTPUT_NAME "solver")
TARGET_LINK_LIBRARIES(structure exadg)

ADD_EXECUTABLE( structure_throughput throughput.cpp)
DEAL_II_SETUP_TARGET(structure_throughput)
set_target_properties(structure_throughput PROPERTIES OUTPUT_NAME "throughput")
TARGET_LINK_LIBRARIES(structure_throughput exadg)
