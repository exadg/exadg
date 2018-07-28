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

IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR
    "\nindexa requires a deal.II installation built with support for p4est but it appears to be missing!\n"
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

include_directories(applications/macros)

# Set the source files to be compiled
SET( TARGET_SRC
     #source/fe_navierstokes_evaluator.cc
     #source/fe_navierstokes_solver.cc
     source/fluid_base_algorithm.cc
     source/SpaldingsLaw.cc
     source/statistics_manager.cc
     source/statistics_manager_ph.cc
     include/solvers_and_preconditioners/multigrid/multigrid_preconditioner_adapter_base.cpp
     include/convection_diffusion/spatial_discretization/operators/convection_operator.cpp
     include/convection_diffusion/spatial_discretization/operators/diffusive_operator.cpp
     include/convection_diffusion/spatial_discretization/operators/mass_operator.cpp
     include/convection_diffusion/spatial_discretization/operators/convection_diffusion_operator.cpp
     include/laplace/spatial_discretization/laplace_operator.cpp
     include/poisson/laplace_operator.cpp
     #include/solvers_and_preconditioners/multigrid/multigrid_preconditioner_adapter_base.cpp
     include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.cpp
     include/solvers_and_preconditioners/mg_coarse_ml/mg_coarse_ml.cpp
     include/solvers_and_preconditioners/mg_coarse_ml/dg_to_cg_transfer.cpp
     include/solvers_and_preconditioners/mg_coarse_ml/mg_coarse_ml_cg.cpp
     include/solvers_and_preconditioners/mg_coarse_ml/mg_coarse_ml_dg.cpp
   )

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include)

PROJECT(indexa)

ADD_LIBRARY(indexa  ${TARGET_SRC})


# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(indexa)

ADD_SUBDIRECTORY(applications)

#ADD_SUBDIRECTORY(xwall_applications)

MESSAGE("====================================================")
