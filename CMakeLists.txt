##
#  CMake script
##

MESSAGE("====================================================")
MESSAGE("=============== Configuring INDEXA =================")
MESSAGE("== High-order incompressible DG towards Exa-scale ==")
MESSAGE("====================================================")


# this is the standard deal.II search mechanism, including check for Trilinos and p4est

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR
    "\nadaflo requires a deal.II installation built with support for Trilinos and p4est but one or both of these appears to be missing!\n"
    )
  IF(NOT DEAL_II_WITH_TRILINOS)
    MESSAGE(FATAL_ERROR
      "\n-- deal.II was built without support for Trilinos!\n"
      )
  ENDIF()

  IF(NOT DEAL_II_WITH_P4EST)
    MESSAGE(FATAL_ERROR
      "\n-- deal.II was built without support for p4est!\n"
      )
  ENDIF()
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set the source files to be compiled
SET( TARGET_SRC
     source/poisson_solver.cc
     source/fluid_base_algorithm.cc
   )

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include)

PROJECT(indexa)

ADD_LIBRARY(indexa SHARED  ${TARGET_SRC})


# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(indexa)

ADD_SUBDIRECTORY(applications)


MESSAGE("====================================================")
