include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  src/main.cc
  )

ADD_EXECUTABLE( "operation-base-vector" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("operation-base-vector")
