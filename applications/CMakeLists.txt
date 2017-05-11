##
#  CMake script
##

PROJECT(indexa_applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     steady_navier_stokes.cc
     unsteady_navier_stokes_dual_splitting.cc
     unsteady_navier_stokes_pressure_correction.cc
     unsteady_navier_stokes_coupled_solver.cc
     unsteady_navier_stokes.cc
     multigrid_solver.cc
     #fe_flow_past_cylinder.cc
     #fe_turbulent_channel.cc
     #fe_poiseuille.cc
     steady_convection_diffusion.cc
     unsteady_convection_diffusion.cc
     #step-7.cc
     laplace_operator_matrix_free.cc
     navier_stokes_operator_matrix_free.cc
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
        TARGET_LINK_LIBRARIES(${testname} indexa)
ENDFOREACH ( sourcefile ${APP_SOURCES} )
