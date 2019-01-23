include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  ../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "p-transfer-2" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("p-transfer-2")