/*
 * momentum_operator.hpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONER_COMPATIBLE_LAPLACE_OPERATOR_HPP_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONER_COMPATIBLE_LAPLACE_OPERATOR_HPP_

#include "compatible_laplace_operator.h"

#include <navierstokes/config.h>

namespace IncNS
{
#if DIM_2 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 1, 0, float>;
#endif
#if DIM_2 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 1, 0, double>;
#endif

#if DIM_2 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 2, 1, float>;
#endif
#if DIM_2 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 2, 1, double>;
#endif

#if DIM_2 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 3, 2, float>;
#endif
#if DIM_2 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 3, 2, double>;
#endif

#if DIM_2 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 4, 3, float>;
#endif
#if DIM_2 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 4, 3, double>;
#endif

#if DIM_2 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 5, 4, float>;
#endif
#if DIM_2 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 5, 4, double>;
#endif

#if DIM_2 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 6, 5, float>;
#endif
#if DIM_2 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 6, 5, double>;
#endif

#if DIM_2 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 7, 6, float>;
#endif
#if DIM_2 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 7, 6, double>;
#endif

#if DIM_2 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 8, 7, float>;
#endif
#if DIM_2 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 8, 7, double>;
#endif

#if DIM_2 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 9, 8, float>;
#endif
#if DIM_2 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 9, 8, double>;
#endif

#if DIM_2 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 10, 9, float>;
#endif
#if DIM_2 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 10, 9, double>;
#endif

#if DIM_2 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 11, 10, float>;
#endif
#if DIM_2 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 11, 10, double>;
#endif

#if DIM_2 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 12, 11, float>;
#endif
#if DIM_2 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 12, 11, double>;
#endif

#if DIM_2 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 13, 12, float>;
#endif
#if DIM_2 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 13, 12, double>;
#endif

#if DIM_2 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 14, 13, float>;
#endif
#if DIM_2 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 14, 13, double>;
#endif

#if DIM_2 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<2, 15, 14, float>;
#endif
#if DIM_2 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<2, 15, 14, double>;
#endif

#if DIM_3 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 1, 0, float>;
#endif
#if DIM_3 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 1, 0, double>;
#endif

#if DIM_3 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 2, 1, float>;
#endif
#if DIM_3 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 2, 1, double>;
#endif

#if DIM_3 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 3, 2, float>;
#endif
#if DIM_3 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 3, 2, double>;
#endif

#if DIM_3 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 4, 3, float>;
#endif
#if DIM_3 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 4, 3, double>;
#endif

#if DIM_3 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 5, 4, float>;
#endif
#if DIM_3 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 5, 4, double>;
#endif

#if DIM_3 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 6, 5, float>;
#endif
#if DIM_3 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 6, 5, double>;
#endif

#if DIM_3 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 7, 6, float>;
#endif
#if DIM_3 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 7, 6, double>;
#endif

#if DIM_3 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 8, 7, float>;
#endif
#if DIM_3 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 8, 7, double>;
#endif

#if DIM_3 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 9, 8, float>;
#endif
#if DIM_3 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 9, 8, double>;
#endif

#if DIM_3 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 10, 9, float>;
#endif
#if DIM_3 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 10, 9, double>;
#endif

#if DIM_3 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 11, 10, float>;
#endif
#if DIM_3 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 11, 10, double>;
#endif

#if DIM_3 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 12, 11, float>;
#endif
#if DIM_3 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 12, 11, double>;
#endif

#if DIM_3 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 13, 12, float>;
#endif
#if DIM_3 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 13, 12, double>;
#endif

#if DIM_3 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 14, 13, float>;
#endif
#if DIM_3 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 14, 13, double>;
#endif

#if DIM_3 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class CompatibleLaplaceOperator<3, 15, 14, float>;
#endif
#if DIM_3 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class CompatibleLaplaceOperator<3, 15, 14, double>;
#endif

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_HPP_ */
