##
#  CMake script
##

PROJECT(applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     # Scalar reaction-convection-diffusion
     convection_diffusion.cc
     convection_diffusion_throughput.cc

     # Incompressible Navier-Stokes
     incompressible_navier_stokes.cc
     incompressible_navier_stokes_throughput.cc
     incompressible_navier_stokes_precursor.cc

     # Incompressible flow with scalar transport
     incompressible_flow_with_transport.cc

     # Compressible Navier-Stokes
     compressible_navier_stokes.cc 
     compressible_navier_stokes_throughput.cc

     # Elasticity
     elasticity.cc
     elasticity_throughput.cc

     # Fluid-structure interaction
     fluid_structure_interaction.cc

     # Other
     gmres_test.cc
     #timer.cc
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
        TARGET_LINK_LIBRARIES(${testname} exadg)
	IF(${USE_DEAL_SPECTRUM})
		target_link_libraries(${testname} ${FFTW3})
	ENDIF()
ENDFOREACH ( sourcefile ${APP_SOURCES} )

IF(${USE_DEAL_SPECTRUM})
	ADD_EXECUTABLE( "deal-spectrum" ../3rdparty/deal.spectrum/src/deal-spectrum.cc)
	DEAL_II_SETUP_TARGET("deal-spectrum")
	TARGET_LINK_LIBRARIES("deal-spectrum" exadg)
ENDIF()

ADD_SUBDIRECTORY(poisson)
