##
#  CMake script
##

PROJECT(applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

ADD_SUBDIRECTORY(compressible_navier_stokes)
ADD_SUBDIRECTORY(convection_diffusion)
ADD_SUBDIRECTORY(fluid_structure_interaction)
ADD_SUBDIRECTORY(incompressible_flow_with_transport)
ADD_SUBDIRECTORY(incompressible_navier_stokes)
ADD_SUBDIRECTORY(poisson)
ADD_SUBDIRECTORY(structure)
