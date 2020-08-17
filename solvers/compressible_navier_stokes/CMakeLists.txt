##
#  CMake script
##

PROJECT(compressible_navier_stokes)

# string replace: cut off .cc from files
ADD_EXECUTABLE( compressible_navier_stokes solver.cpp)
DEAL_II_SETUP_TARGET(compressible_navier_stokes)
set_target_properties(compressible_navier_stokes PROPERTIES OUTPUT_NAME "solver")
TARGET_LINK_LIBRARIES(compressible_navier_stokes exadg)

ADD_EXECUTABLE( compressible_navier_stokes_throughput throughput.cpp)
DEAL_II_SETUP_TARGET(compressible_navier_stokes_throughput)
set_target_properties(compressible_navier_stokes_throughput PROPERTIES OUTPUT_NAME "throughput")
TARGET_LINK_LIBRARIES(compressible_navier_stokes_throughput exadg)
