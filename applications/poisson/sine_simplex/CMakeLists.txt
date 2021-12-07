PROJECT(poisson_sine)

EXADG_PICKUP_EXE(solver.cpp poisson_sine_simplex solver)

ADD_SUBDIRECTORY(tests)
