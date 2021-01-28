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
        STRING( REPLACE ".cpp" "" targetname ${sourcefile} )
        EXADG_PICKUP_EXE(${sourcefile} ${targetname} ${targetname})
ENDFOREACH ()