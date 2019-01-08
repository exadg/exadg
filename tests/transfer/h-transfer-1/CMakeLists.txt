include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  src/main.cc
  )

ADD_EXECUTABLE( "h-transfer-1" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("h-transfer-1")
