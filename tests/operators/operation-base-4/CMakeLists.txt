include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "operation-base-4" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("operation-base-4")