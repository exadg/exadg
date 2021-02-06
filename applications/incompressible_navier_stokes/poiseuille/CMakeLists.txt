PROJECT(incompressible_poiseuille)

EXADG_PICKUP_EXE(solver.cpp incompressible_poiseuille solver)

ADD_SUBDIRECTORY(tests)