#########################################################################
# 
#                 #######               ######  #######
#                 ##                    ##   ## ##
#                 #####   ##  ## #####  ##   ## ## ####
#                 ##       ####  ## ##  ##   ## ##   ##
#                 ####### ##  ## ###### ######  #######
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2021 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#########################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

SET(CMAKE_CXX_STANDARD 17)

ADD_SUBDIRECTORY(cmake)

PRINT_EXADG_HEADER()

# deal.II
FIND_PACKAGE(deal.II 9.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# p4est
IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR
    "\nExaDG requires a deal.II installation built with support for p4est but it appears to be missing!\n"
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Add info of git branch 
DEAL_II_QUERY_GIT_INFORMATION(EXADG)
CONFIGURE_FILE(
  ${CMAKE_SOURCE_DIR}/include/exadg/utilities/exadg_revision.h.in
  ${CMAKE_SOURCE_DIR}/include/exadg/utilities/exadg_revision.h
  )


# Set the source files to be compiled
SET(TARGET_SRC
     include/exadg/utilities/timer_tree.cpp
     include/exadg/utilities/print_general_infos.cpp
     include/exadg/time_integration/bdf_constants.cpp
     include/exadg/time_integration/ab_constants.cpp
     include/exadg/time_integration/am_constants.cpp
     include/exadg/time_integration/extrapolation_constants.cpp
     include/exadg/time_integration/time_int_base.cpp
     include/exadg/time_integration/time_int_multistep_base.cpp
     include/exadg/time_integration/time_int_bdf_base.cpp
     include/exadg/time_integration/time_int_explicit_runge_kutta_base.cpp
     include/exadg/time_integration/time_int_gen_alpha_base.cpp
     include/exadg/functions_and_boundary_conditions/container_interface_data.cpp
     include/exadg/functions_and_boundary_conditions/linear_interpolation.cpp
     include/exadg/functions_and_boundary_conditions/interface_coupling.cpp
     include/exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.cpp
     include/exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.cpp
     include/exadg/solvers_and_preconditioners/multigrid/transfer.cpp
     include/exadg/postprocessor/time_control.cpp
     include/exadg/postprocessor/time_control_statistics.cpp
     include/exadg/postprocessor/error_calculation.cpp
     include/exadg/postprocessor/mean_scalar_calculation.cpp
     include/exadg/postprocessor/normal_flux_calculation.cpp
     include/exadg/postprocessor/lift_and_drag_calculation.cpp
     include/exadg/postprocessor/pressure_difference_calculation.cpp
     include/exadg/postprocessor/kinetic_energy_spectrum.cpp
     include/exadg/postprocessor/kinetic_energy_calculation.cpp
     include/exadg/postprocessor/statistics_manager.cpp
     include/exadg/operators/operator_base.cpp
     include/exadg/operators/mass_operator.cpp
     include/exadg/operators/rhs_operator.cpp
     include/exadg/operators/navier_stokes_calculators.cpp
     # Poisson equation
     include/exadg/poisson/user_interface/parameters.cpp
     include/exadg/poisson/spatial_discretization/operator.cpp
     include/exadg/poisson/spatial_discretization/laplace_operator.cpp
     include/exadg/poisson/preconditioners/multigrid_preconditioner.cpp
     include/exadg/poisson/postprocessor/postprocessor.cpp
     include/exadg/poisson/driver.cpp
     include/exadg/poisson/overset_grids/driver.cpp
     # convection-diffusion equation
     include/exadg/convection_diffusion/user_interface/parameters.cpp
     include/exadg/convection_diffusion/spatial_discretization/operators/convective_operator.cpp
     include/exadg/convection_diffusion/spatial_discretization/operators/diffusive_operator.cpp
     include/exadg/convection_diffusion/spatial_discretization/operators/combined_operator.cpp
     include/exadg/convection_diffusion/spatial_discretization/operator.cpp
     include/exadg/convection_diffusion/spatial_discretization/interface.cpp
     include/exadg/convection_diffusion/preconditioners/multigrid_preconditioner.cpp
     include/exadg/convection_diffusion/time_integration/time_int_bdf.cpp
     include/exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.cpp
     include/exadg/convection_diffusion/time_integration/driver_steady_problems.cpp
     include/exadg/convection_diffusion/postprocessor/postprocessor.cpp
     include/exadg/postprocessor/output_generator_scalar.cpp
     include/exadg/postprocessor/pointwise_output_generator_base.cpp
     include/exadg/convection_diffusion/driver.cpp
     # incompressible Navier-Stokes equations
     include/exadg/incompressible_navier_stokes/user_interface/parameters.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/convective_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/gradient_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/divergence_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/rhs_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/divergence_penalty_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/continuity_penalty_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operators/projection_operator.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model_base.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/turbulence_model.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/generalized_newtonian_model.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.cpp
     include/exadg/incompressible_navier_stokes/spatial_discretization/calculators/streamfunction_calculator_rhs_operator.cpp
     include/exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.cpp
     include/exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_projection.cpp
     include/exadg/incompressible_navier_stokes/time_integration/time_int_bdf.cpp
     include/exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.cpp
     include/exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.cpp
     include/exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.cpp
     include/exadg/incompressible_navier_stokes/time_integration/driver_steady_problems.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/output_generator.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/pointwise_output_generator.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/divergence_and_mass_error.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/inflow_data_calculator.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/kinetic_energy_dissipation_detailed.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/flow_rate_calculator.cpp
     include/exadg/incompressible_navier_stokes/postprocessor/postprocessor.cpp
     include/exadg/incompressible_navier_stokes/driver.cpp
     include/exadg/incompressible_navier_stokes/precursor/driver.cpp
     # incompressible flow with transport
     include/exadg/incompressible_flow_with_transport/driver.cpp
     # compressible Navier-Stokes equations
     include/exadg/compressible_navier_stokes/user_interface/parameters.cpp
     include/exadg/compressible_navier_stokes/time_integration/time_int_explicit_runge_kutta.cpp
     include/exadg/compressible_navier_stokes/spatial_discretization/interface.cpp
     include/exadg/compressible_navier_stokes/spatial_discretization/operator.cpp
     include/exadg/compressible_navier_stokes/postprocessor/output_generator.cpp
     include/exadg/compressible_navier_stokes/postprocessor/pointwise_output_generator.cpp
     include/exadg/compressible_navier_stokes/postprocessor/postprocessor.cpp
     include/exadg/compressible_navier_stokes/driver.cpp
     # elasticity
     include/exadg/structure/user_interface/parameters.cpp
     include/exadg/structure/material/library/st_venant_kirchhoff.cpp
     include/exadg/structure/spatial_discretization/operators/elasticity_operator_base.cpp
     include/exadg/structure/spatial_discretization/operators/nonlinear_operator.cpp
     include/exadg/structure/spatial_discretization/operators/linear_operator.cpp
     include/exadg/structure/spatial_discretization/operators/body_force_operator.cpp
     include/exadg/structure/spatial_discretization/operator.cpp
     include/exadg/structure/spatial_discretization/interface.cpp
     include/exadg/structure/preconditioners/multigrid_preconditioner.cpp
     include/exadg/structure/time_integration/driver_steady_problems.cpp
     include/exadg/structure/time_integration/driver_quasi_static_problems.cpp
     include/exadg/structure/time_integration/time_int_gen_alpha.cpp
     include/exadg/structure/postprocessor/output_generator.cpp
     include/exadg/structure/postprocessor/postprocessor.cpp
     include/exadg/structure/driver.cpp
     # fluid-structure-interaction
     include/exadg/fluid_structure_interaction/driver.cpp
     # acoustic conservation equations
     include/exadg/acoustic_conservation_equations/user_interface/parameters.cpp
     include/exadg/acoustic_conservation_equations/spatial_discretization/spatial_operator.cpp
     include/exadg/acoustic_conservation_equations/postprocessor/pointwise_output_generator.cpp
     include/exadg/acoustic_conservation_equations/postprocessor/postprocessor.cpp
     include/exadg/acoustic_conservation_equations/postprocessor/output_generator.cpp
     include/exadg/acoustic_conservation_equations/driver.cpp
     )

SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)

# Set the name of the project
PROJECT(exadg)

OPTION(PICKUP_TESTS "Pick up ExaDG tests." ON)
OPTION(BUILD_SHARED_LIBS "Build shared library." ON)

ADD_LIBRARY(exadg ${TARGET_SRC})

# Set the include directories
SET(EXADG_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/bundled)
TARGET_INCLUDE_DIRECTORIES(exadg PUBLIC ${EXADG_INCLUDE_DIRS})

# FFTW
OPTION(EXADG_WITH_FFTW "Use FFTW" OFF)
IF(${EXADG_WITH_FFTW})
    ADD_DEFINITIONS(-DEXADG_WITH_FFTW)
    INCLUDE_DIRECTORIES(${FFTW_INCLUDE})
    IF(${BUILD_SHARED_LIBS})
        FIND_LIBRARY(FFTW3 fftw3 HINTS ${FFTW_LIB})
        FIND_LIBRARY(FFTW3_MPI fftw3_mpi HINTS ${FFTW_LIB})
    ELSE()
        FIND_LIBRARY(FFTW3 libfftw3.a HINTS ${FFTW_LIB})
        FIND_LIBRARY(FFTW3_MPI libfftw3_mpi.a HINTS ${FFTW_LIB})
    ENDIF()
    MESSAGE(STATUS "Found FFTW lib at ${FFTW3}")
    MESSAGE(STATUS "Found FFTW mpi lib at ${FFTW3_MPI}")
    TARGET_LINK_FFTW(exadg)
ENDIF()

# LIKWID
OPTION(EXADG_WITH_LIKDWID "Use LIKWID" OFF)
IF(${EXADG_WITH_LIKWID})
    ADD_DEFINITIONS(-DEXADG_WITH_LIKWID)
    FIND_LIBRARY(LIKWID likwid HINTS ${LIKWID_LIB})
    MESSAGE(STATUS "Found LIKWID lib at ${LIKWID}")
    # The macro LIKWID_PERFMON is used inside Likwid, i.e., we need to pass
    # that precise macro into the Likwid machinery for compilation.
    TARGET_COMPILE_DEFINITIONS(exadg PUBLIC LIKWID_PERFMON)
    TARGET_INCLUDE_DIRECTORIES(exadg PUBLIC ${LIKWID_INCLUDE})
    TARGET_LINK_LIBRARIES(exadg ${LIKWID})
ENDIF()

# preCICE
OPTION(EXADG_WITH_PRECICE "Use preCICE" OFF})
IF(${EXADG_WITH_PRECICE})
    # the environment variable precice_DIR is searched by default
    FIND_PACKAGE(precice 2.0
      HINTS ${precice_DIR} ${PRECICE_DIR} $ENV{PRECICE_DIR}
    )
    IF(NOT ${precice_FOUND})
      MESSAGE(FATAL_ERROR "\n"
       "*** Could not locate a (sufficiently recent) version of preCICE. ***\n\n"
       "You may want to either pass a flag -Dprecice_DIR=/path/to/precice to cmake\n"
       "(where the path points to the installation prefix or the build directory)\n"
       " or set an environment variable \"precice_DIR\" that contains this path.")
    ENDIF()
    ADD_DEFINITIONS(-DEXADG_WITH_PRECICE)
    MESSAGE(STATUS "Found preCICE version ${precice_VERSION} at ${precice_DIR}")
    TARGET_LINK_LIBRARIES(exadg precice::precice)
ENDIF()

# DEBUG vs. RELEASE
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(exadg)

ENABLE_TESTING()

ADD_SUBDIRECTORY(applications)
ADD_SUBDIRECTORY(prototypes)
ADD_SUBDIRECTORY(tests)

ADD_SUBDIRECTORY(doc/doxygen)

####################################################
# Export EXADG for easy interfacing: After a FIND_PACKAGE call in CMake you will have access to
# EXADG_CMAKE_DIR, EXADG_INCLUDE_DIRS and EXADG_VERSION (which might be helpful in the future
# once EXADG has a version number).

INCLUDE(CMakePackageConfigHelpers)
SET(EXADG_MACRO_DIR ${CMAKE_SOURCE_DIR}/cmake/macros)

# EXADGConfig.cmake.in serves as a template. Running CMake will create EXADGConfig.cmake and
# EXADGTargets.cmake from it in the build folder. These files are needed to use the
# FIND_PACKAGE command in other projects.
CONFIGURE_PACKAGE_CONFIG_FILE(EXADGConfig.cmake.in EXADGConfig.cmake
  INSTALL_DESTINATION .
  PATH_VARS 
  EXADG_INCLUDE_DIRS
  EXADG_MACRO_DIR
  )

# Within EXADGConfig.cmake we provide the options of the current build.
# This is done by including the file EXADGOptions.cmake in which the options are stored.
CONFIGURE_FILE(EXADGOptions.cmake.in EXADGOptions.cmake @ONLY)

EXPORT(TARGETS
  exadg
  NAMESPACE EXADG::
  FILE EXADGTargets.cmake
  )

SET(CMAKE_EXPORT_PACKAGE_REGISTRY ON)

EXPORT(PACKAGE EXADG)

# To link against EXADG use:
# FIND_PACKAGE(EXADG REQUIRED HINTS ${EXADG_DIR})
# TARGET_LINK_LIBRARIES(<target_which_links_to_exadg> EXADG::exadg)
# To create an executable use:
# EXADG_PICKUP_EXE(<file> <target> <targetname>)

