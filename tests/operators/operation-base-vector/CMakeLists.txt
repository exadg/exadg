include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  src/main.cc
  ../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.cpp
  ../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_c.cpp
  ../../../include/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.cpp
  )

ADD_EXECUTABLE( "operation-base-vector" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("operation-base-vector")
