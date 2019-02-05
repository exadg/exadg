include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  ../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.cpp
  ../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_c.cpp
  ../../../include/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "p-transfer-3" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("p-transfer-3")
