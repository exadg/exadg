include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  src/main.cc
  )

ADD_EXECUTABLE( "performance-stream" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("performance-stream")


# load likwid
add_definitions(-DLIKWID_PERFMON)
include_directories("$ENV{LIKWID_BASE}/include")
FIND_LIBRARY(LIKWID likwid HINTS "$ENV{LIKWID_BASE}/lib")
TARGET_LINK_LIBRARIES("performance-stream" ${LIKWID})
