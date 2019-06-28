include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  ../../../include/solvers_and_preconditioners/multigrid/transfer/mg_transfer_mf_c.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "dg-to-cg-transfer-3" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("dg-to-cg-transfer-3")
