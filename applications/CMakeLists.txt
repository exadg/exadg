##
#  CMake script
##

PROJECT(applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     # Compressible Navier-Stokes
     compressible_navier_stokes_operator_matrix_free.cc 
     unsteady_compressible_navier_stokes.cc
     
     # Incompressible Navier-Stokes
     navier_stokes_operator_matrix_free.cc
     steady_navier_stokes.cc
     unsteady_navier_stokes.cc
     unsteady_navier_stokes_two_domains.cc
     
     # Poisson
     poisson.cc
     poisson_dual.cc
     laplace_operator_matrix_free.cc
     poisson_lung.cc

     lung_tria.cc
     lung_test.cc

     #multigrid_solver.cc

     # Scalar reaction-convection-diffusion
     steady_convection_diffusion.cc
     unsteady_convection_diffusion.cc

     # incompressible flow with scalar transport
     unsteady_incompressible_flow_with_transport.cc

     # Other
     gmres_test.cc
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
        TARGET_LINK_LIBRARIES(${testname} exadg)

        option(USE_DEAL_SPECTRUM "Use deal.spectrum" OFF)
        IF(${USE_DEAL_SPECTRUM})
            include_directories(${FFTW_INC})
            FIND_LIBRARY(FFTW3 fftw3 HINTS ${FFTW_LIB})
            add_definitions(-DUSE_DEAL_SPECTRUM)
            MESSAGE(${FFTW3})
            target_link_libraries(${testname} ${FFTW3})
        ENDIF()

ENDFOREACH ( sourcefile ${APP_SOURCES} )
