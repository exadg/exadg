include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/solvers_and_preconditioners/multigrid/transfer/mg_transfer_mf_p.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "p-transfer-4" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("p-transfer-4")
