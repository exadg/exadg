include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "performance-operators" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("performance-operators")
