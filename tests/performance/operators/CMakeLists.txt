include_directories(BEFORE include)
include_directories(BEFORE ../../../)

SET(TARGET_SRC
  ../../../include/poisson/spatial_discretization/laplace_operator.cpp
#  ../../../include/convection_diffusion/spatial_discretization/operators/mass_matrix_operator.cpp
  ../../../include/convection_diffusion/spatial_discretization/operators/diffusive_operator.cpp
  ../../../include/convection_diffusion/spatial_discretization/operators/convective_operator.cpp
  src/main.cc
  )

ADD_EXECUTABLE( "performance-operators" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("performance-operators")


# load likwid
FIND_LIBRARY(LIKWID likwid HINTS "$ENV{LIKWID_BASE}/lib")

IF(NOT LIKWID)
    MESSAGE("performance-operators is compiled without LIKWID!")
ELSE()
    MESSAGE("performance-operators is compiled with LIKWID!")
    add_definitions(-DLIKWID_PERFMON)
    include_directories("$ENV{LIKWID_BASE}/include")
    TARGET_LINK_LIBRARIES("performance-operators" ${LIKWID})
ENDIF()
