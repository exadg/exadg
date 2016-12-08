##
#  CMake script
##

PROJECT(indexa_applications)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     SteadyNavierStokes.cc
     UnsteadyNavierStokesDualSplitting.cc
     UnsteadyNavierStokesPressureCorrection.cc
     UnsteadyNavierStokesCoupledSolver.cc
     multigrid_solver.cc
     #fe_flow_past_cylinder.cc
     #fe_turbulent_channel.cc
     #fe_poiseuille.cc
     SteadyConvectionDiffusion.cc
     UnsteadyConvectionDiffusion.cc
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
        TARGET_LINK_LIBRARIES(${testname} indexa)
ENDFOREACH ( sourcefile ${APP_SOURCES} )
