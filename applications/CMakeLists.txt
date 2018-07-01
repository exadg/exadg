##
#  CMake script
##

PROJECT(indexa_applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     #compressible_navier_stokes_operator_matrix_free.cc 
     #steady_navier_stokes.cc
     #unsteady_navier_stokes.cc
     #unsteady_navier_stokes_two_domains.cc
     #multigrid_solver.cc
     #fe_flow_past_cylinder.cc
     #fe_turbulent_channel.cc
     #fe_poiseuille.cc
     steady_convection_diffusion.cc
     #unsteady_convection_diffusion.cc
     #step-7.cc
     #laplace_operator_matrix_free.cc
     #navier_stokes_operator_matrix_free.cc
     #navier_stokes_operator_matrix_free_basic.cc
     #gmres_test.cc
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
        TARGET_LINK_LIBRARIES(${testname} indexa)

        option(USE_DEAL_SPECTRUM "Use deal.spectrum" OFF)
        IF(${USE_DEAL_SPECTRUM})
            include_directories(${FFTW_INC})
            FIND_LIBRARY(FFTW3 fftw3_mpi HINTS ${FFTW_LIB})
            add_definitions(-DUSE_DEAL_SPECTRUM)
            MESSAGE(${FFTW3})
            target_link_libraries(${testname} ${FFTW3})
        ENDIF()

ENDFOREACH ( sourcefile ${APP_SOURCES} )
