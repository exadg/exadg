##
#  CMake script for the step-37 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "main")

include_directories(include)

# Declare all source files the target consists of:
SET(TARGET_SRC
  ../../operators/operation-base-util/laplace_operator.cpp
  ../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.cpp
  src/main.cc
  # You can specify additional files here!
  )

#include_directories(${FFTW_INC})
#add_library(${FFTW_LIB})

#add_definitions( ${FFTW_INC} )
#add_definitions( ${FFTW_LIB} )

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FFTW_INC} ${FFTW_LIB}")

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fullfilled?
#
#IF(NOT DEAL_II_WITH_LAPACK)
#  MESSAGE(FATAL_ERROR "
#Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
#    DEAL_II_WITH_LAPACK = ON
#One or all of these are OFF in your installation but are required for this tutorial step."
#    )
#ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

# load likwid
add_definitions(-DLIKWID_PERFMON)
include_directories("$ENV{LIKWID_BASE}/include")
FIND_LIBRARY(LIKWID likwid HINTS "$ENV{LIKWID_BASE}/lib")
TARGET_LINK_LIBRARIES(${TARGET} ${LIKWID})