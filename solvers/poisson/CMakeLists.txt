##
#  CMake script
##

PROJECT(poisson)

ADD_EXECUTABLE( poisson solver.cpp)
DEAL_II_SETUP_TARGET(poisson)
set_target_properties(poisson PROPERTIES OUTPUT_NAME "solver")
TARGET_LINK_LIBRARIES(poisson exadg)

ADD_EXECUTABLE( poisson_throughput throughput.cpp)
DEAL_II_SETUP_TARGET(poisson_throughput)
set_target_properties(poisson_throughput PROPERTIES OUTPUT_NAME "throughput")
TARGET_LINK_LIBRARIES(poisson_throughput exadg)

ADD_SUBDIRECTORY(tests)
