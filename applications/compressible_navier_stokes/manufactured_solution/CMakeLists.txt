PROJECT(compressible_manufactured)

EXADG_PICKUP_EXE(solver.cpp compressible_manufactured solver)

ADD_SUBDIRECTORY(tests)