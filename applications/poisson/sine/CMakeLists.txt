PROJECT(poisson_sine)

EXADG_PICKUP_EXE(solver.cpp poisson_sine solver)

ADD_SUBDIRECTORY(tests)
