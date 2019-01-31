include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "performance-laplace" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("performance-laplace")


# load likwid
FIND_LIBRARY(LIKWID likwid HINTS "$ENV{LIKWID_BASE}/lib")

IF(NOT LIKWID)
    MESSAGE("performance-laplace is compiled without LIKWID!")
ELSE()
    MESSAGE("performance-laplace is compiled with LIKWID!")
    add_definitions(-DLIKWID_PERFMON)
    include_directories("$ENV{LIKWID_BASE}/include")
    TARGET_LINK_LIBRARIES("performance-laplace" ${LIKWID})
ENDIF()