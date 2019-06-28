include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  src/main.cc
  ../../../include/solvers_and_preconditioners/multigrid/transfer/mg_transfer_mf_p.cpp
  ../../../include/solvers_and_preconditioners/multigrid/transfer/mg_transfer_mf_c.cpp
  )

ADD_EXECUTABLE( "h-transfer-1" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("h-transfer-1")
