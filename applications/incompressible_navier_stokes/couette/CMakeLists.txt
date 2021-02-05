PROJECT(incompressible_couette)

EXADG_PICKUP_EXE(solver.cpp incompressible_couette solver)

ADD_SUBDIRECTORY(tests)