PROJECT(tests)

SET( SOURCE_FILES
     operators/operation-base-1
     operators/operation-base-2
     operators/operation-base-3
     operators/operation-base-4
     transfer/dg-to-cg-transfer-1
     transfer/dg-to-cg-transfer-2
     transfer/dg-to-cg-transfer-3
     transfer/p-transfer-1
     transfer/p-transfer-2
     transfer/p-transfer-3
     performance/operators
     performance/distance_computer
     )

FOREACH ( sourcefile ${SOURCE_FILES} )

    ADD_SUBDIRECTORY(${sourcefile})

ENDFOREACH ( sourcefile ${APP_SOURCES} )
