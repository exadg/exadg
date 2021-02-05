PROJECT(conv_diff_boundary_layer)

EXADG_PICKUP_EXE(solver.cpp conv_diff_boundary_layer solver)

ADD_SUBDIRECTORY(tests)
