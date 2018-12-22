include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  ../../../include/convection_diffusion/spatial_discretization/operators/mass_operator.cpp
  ../../../include/convection_diffusion/spatial_discretization/operators/diffusive_operator.cpp
  ../../../include/convection_diffusion/spatial_discretization/operators/convective_operator.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "performance-operators" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("performance-operators")
