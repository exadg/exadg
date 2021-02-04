PROJECT(structure_manufactured)

EXADG_PICKUP_EXE(solver.cpp structure_manufactured solver)
EXADG_PICKUP_EXE(throughput.cpp structure_throughput throughput)

ADD_SUBDIRECTORY(tests)