##
#  CMake script
##

PROJECT(prototypes)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     moving_mesh.cpp
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cpp from files
        STRING( REPLACE ".cpp" "" testname ${sourcefile} )
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
