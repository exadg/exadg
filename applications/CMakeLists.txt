##
#  CMake script
##

PROJECT(indexa_applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     #steady_navier_stokes.cc
     unsteady_navier_stokes.cc
     #unsteady_navier_stokes_two_domains.cc
     #multigrid_solver.cc
     #fe_flow_past_cylinder.cc
     #fe_turbulent_channel.cc
     #fe_poiseuille.cc
     #steady_convection_diffusion.cc
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

        include_directories(/lrz/sys/libraries/fftw/3.3.3/sse/include)

        FIND_LIBRARY(LIKWID_FFTW3_1 fftw3_mpi HINTS /lrz/sys/libraries/fftw/3.3.3/intelmpi/lib)
        FIND_LIBRARY(LIKWID_FFTW3_2 fftw3 HINTS /lrz/sys/libraries/fftw/3.3.3/sse/lib)

        MESSAGE(${LIKWID_FFTW3_1})
        MESSAGE(${LIKWID_FFTW3_2})

        target_link_libraries(${testname} ${LIKWID_FFTW3_2} ${LIKWID_FFTW3_1})

ENDFOREACH ( sourcefile ${APP_SOURCES} )
