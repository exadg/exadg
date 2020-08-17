##
#  CMake script
##

PROJECT(fluid_structure_interaction)

# string replace: cut off .cc from files
ADD_EXECUTABLE( fluid_structure_interaction solver.cpp)
DEAL_II_SETUP_TARGET(fluid_structure_interaction)
set_target_properties(fluid_structure_interaction PROPERTIES OUTPUT_NAME "solver")
TARGET_LINK_LIBRARIES(fluid_structure_interaction exadg)
