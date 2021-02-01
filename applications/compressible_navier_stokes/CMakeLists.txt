##
#  CMake script
##

PROJECT(compressible_navier_stokes)

ADD_SUBDIRECTORY(couette)
ADD_SUBDIRECTORY(euler_vortex)
ADD_SUBDIRECTORY(manufactured_solution)
ADD_SUBDIRECTORY(poiseuille)
ADD_SUBDIRECTORY(shear_flow)
ADD_SUBDIRECTORY(taylor_green)
ADD_SUBDIRECTORY(template)
ADD_SUBDIRECTORY(turbulent_channel)