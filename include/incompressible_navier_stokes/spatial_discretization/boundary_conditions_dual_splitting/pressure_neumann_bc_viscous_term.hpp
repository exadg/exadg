/*
 * pressure_neumann_bc_viscous_term.hpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_VISCOUS_TERM_HPP_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_VISCOUS_TERM_HPP_


#include "pressure_neumann_bc_viscous_term.h"

#include <navierstokes/config.h>

namespace IncNS
{
/*
 * dim = 2
 */

#if DIM_2 && DEGREE_1 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 1, 0, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 1, 1, float>;
#  endif
#endif
#if DIM_2 && DEGREE_1 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 1, 0, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 1, 1, double>;
#  endif
#endif

#if DIM_2 && DEGREE_2 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 2, 1, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 2, 2, float>;
#  endif
#endif
#if DIM_2 && DEGREE_2 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 2, 1, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 2, 2, double>;
#  endif
#endif

#if DIM_2 && DEGREE_3 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 3, 2, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 3, 3, float>;
#  endif
#endif
#if DIM_2 && DEGREE_3 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 3, 2, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 3, 3, double>;
#  endif
#endif

#if DIM_2 && DEGREE_4 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 4, 3, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 4, 4, float>;
#  endif
#endif
#if DIM_2 && DEGREE_4 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 4, 3, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 4, 4, double>;
#  endif
#endif

#if DIM_2 && DEGREE_5 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 5, 4, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 5, 5, float>;
#  endif
#endif
#if DIM_2 && DEGREE_5 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 5, 4, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 5, 5, double>;
#  endif
#endif

#if DIM_2 && DEGREE_6 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 6, 5, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 6, 6, float>;
#  endif
#endif
#if DIM_2 && DEGREE_6 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 6, 5, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 6, 6, double>;
#  endif
#endif

#if DIM_2 && DEGREE_7 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 7, 6, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 7, 7, float>;
#  endif
#endif
#if DIM_2 && DEGREE_7 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 7, 6, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 7, 7, double>;
#  endif
#endif

#if DIM_2 && DEGREE_8 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 8, 7, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 8, 8, float>;
#  endif
#endif
#if DIM_2 && DEGREE_8 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 8, 7, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 8, 8, double>;
#  endif
#endif

#if DIM_2 && DEGREE_9 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 9, 8, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 9, 9, float>;
#  endif
#endif
#if DIM_2 && DEGREE_9 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 9, 8, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 9, 9, double>;
#  endif
#endif

#if DIM_2 && DEGREE_10 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 10, 9, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 10, 10, float>;
#  endif
#endif
#if DIM_2 && DEGREE_10 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 10, 9, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 10, 10, double>;
#  endif
#endif

#if DIM_2 && DEGREE_11 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 11, 10, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 11, 11, float>;
#  endif
#endif
#if DIM_2 && DEGREE_11 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 11, 10, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 11, 11, double>;
#  endif
#endif

#if DIM_2 && DEGREE_12 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 12, 11, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 12, 12, float>;
#  endif
#endif
#if DIM_2 && DEGREE_12 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 12, 11, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 12, 12, double>;
#  endif
#endif

#if DIM_2 && DEGREE_13 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 13, 12, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 13, 13, float>;
#  endif
#endif
#if DIM_2 && DEGREE_13 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 13, 12, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 13, 13, double>;
#  endif
#endif

#if DIM_2 && DEGREE_14 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 14, 13, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 14, 14, float>;
#  endif
#endif
#if DIM_2 && DEGREE_14 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 14, 13, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 14, 14, double>;
#  endif
#endif

#if DIM_2 && DEGREE_15 && OP_FLOAT
template class PressureNeumannBCViscousTerm<2, 15, 14, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 15, 15, float>;
#  endif
#endif
#if DIM_2 && DEGREE_15 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<2, 15, 14, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<2, 15, 15, double>;
#  endif
#endif

/*
 * dim = 3
 */

#if DIM_3 && DEGREE_1 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 1, 0, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 1, 1, float>;
#  endif
#endif
#if DIM_3 && DEGREE_1 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 1, 0, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 1, 1, double>;
#  endif
#endif

#if DIM_3 && DEGREE_2 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 2, 1, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 2, 2, float>;
#  endif
#endif
#if DIM_3 && DEGREE_2 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 2, 1, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 2, 2, double>;
#  endif
#endif

#if DIM_3 && DEGREE_3 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 3, 2, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 3, 3, float>;
#  endif
#endif
#if DIM_3 && DEGREE_3 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 3, 2, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 3, 3, double>;
#  endif
#endif

#if DIM_3 && DEGREE_4 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 4, 3, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 4, 4, float>;
#  endif
#endif
#if DIM_3 && DEGREE_4 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 4, 3, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 4, 4, double>;
#  endif
#endif

#if DIM_3 && DEGREE_5 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 5, 4, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 5, 5, float>;
#  endif
#endif
#if DIM_3 && DEGREE_5 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 5, 4, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 5, 5, double>;
#  endif
#endif

#if DIM_3 && DEGREE_6 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 6, 5, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 6, 6, float>;
#  endif
#endif
#if DIM_3 && DEGREE_6 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 6, 5, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 6, 6, double>;
#  endif
#endif

#if DIM_3 && DEGREE_7 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 7, 6, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 7, 7, float>;
#  endif
#endif
#if DIM_3 && DEGREE_7 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 7, 6, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 7, 7, double>;
#  endif
#endif

#if DIM_3 && DEGREE_8 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 8, 7, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 8, 8, float>;
#  endif
#endif
#if DIM_3 && DEGREE_8 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 8, 7, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 8, 8, double>;
#  endif
#endif

#if DIM_3 && DEGREE_9 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 9, 8, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 9, 9, float>;
#  endif
#endif
#if DIM_3 && DEGREE_9 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 9, 8, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 9, 9, double>;
#  endif
#endif

#if DIM_3 && DEGREE_10 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 10, 9, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 10, 10, float>;
#  endif
#endif
#if DIM_3 && DEGREE_10 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 10, 9, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 10, 10, double>;
#  endif
#endif

#if DIM_3 && DEGREE_11 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 11, 10, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 11, 11, float>;
#  endif
#endif
#if DIM_3 && DEGREE_11 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 11, 10, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 11, 11, double>;
#  endif
#endif

#if DIM_3 && DEGREE_12 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 12, 11, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 12, 12, float>;
#  endif
#endif
#if DIM_3 && DEGREE_12 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 12, 11, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 12, 12, double>;
#  endif
#endif

#if DIM_3 && DEGREE_13 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 13, 12, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 13, 13, float>;
#  endif
#endif
#if DIM_3 && DEGREE_13 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 13, 12, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 13, 13, double>;
#  endif
#endif

#if DIM_3 && DEGREE_14 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 14, 13, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 14, 14, float>;
#  endif
#endif
#if DIM_3 && DEGREE_14 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 14, 13, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 14, 14, double>;
#  endif
#endif

#if DIM_3 && DEGREE_15 && OP_FLOAT
template class PressureNeumannBCViscousTerm<3, 15, 14, float>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 15, 15, float>;
#  endif
#endif
#if DIM_3 && DEGREE_15 && OP_DOUBLE
template class PressureNeumannBCViscousTerm<3, 15, 14, double>;
#  if EQUAL_ORDER
template class PressureNeumannBCViscousTerm<3, 15, 15, double>;
#  endif
#endif

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_VISCOUS_TERM_HPP_ \
        */
