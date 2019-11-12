##
#  CMake script
##

PROJECT(applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     # Poisson
     poisson.cc
     #poisson_old.cc
     #poisson_dual.cc
     #multigrid_solver.cc
     #poisson_lung.cc
     #lung_tria.cc
     #laplace_operator_matrix_free.cc
     poisson_performance.cc
 

     # Scalar reaction-convection-diffusion
     convection_diffusion_performance.cc
     convection_diffusion.cc

     # Incompressible Navier-Stokes
     incompressible_navier_stokes_performance.cc
     incompressible_navier_stokes.cc
     incompressible_navier_stokes_two_domains.cc

     # incompressible flow with scalar transport
     incompressible_flow_with_transport.cc

     # Compressible Navier-Stokes
     compressible_navier_stokes_performance.cc 
     compressible_navier_stokes.cc

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
            include_directories($ENV{FFTW_INC})
            FIND_LIBRARY(FFTW3 fftw3 HINTS $ENV{FFTW_LIB})
            add_definitions(-DUSE_DEAL_SPECTRUM)
            MESSAGE(${FFTW3})
            target_link_libraries(${testname} ${FFTW3})
        ENDIF()

ENDFOREACH ( sourcefile ${APP_SOURCES} )
