/*
 * inverse_mass_matrix.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "inverse_mass_matrix.h"

#include <navierstokes/config.h>

#if DIM_2 && DEGREE_0 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 0, double, 1>;
template class InverseMassMatrixOperator<2, 0, double, 2>;
template class InverseMassMatrixOperator<2, 0, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_1 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 1, double, 1>;
template class InverseMassMatrixOperator<2, 1, double, 2>;
template class InverseMassMatrixOperator<2, 1, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_2 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 2, double, 1>;
template class InverseMassMatrixOperator<2, 2, double, 2>;
template class InverseMassMatrixOperator<2, 2, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_3 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 3, double, 1>;
template class InverseMassMatrixOperator<2, 3, double, 2>;
template class InverseMassMatrixOperator<2, 3, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_4 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 4, double, 1>;
template class InverseMassMatrixOperator<2, 4, double, 2>;
template class InverseMassMatrixOperator<2, 4, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_5 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 5, double, 1>;
template class InverseMassMatrixOperator<2, 5, double, 2>;
template class InverseMassMatrixOperator<2, 5, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_6 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 6, double, 1>;
template class InverseMassMatrixOperator<2, 6, double, 2>;
template class InverseMassMatrixOperator<2, 6, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_7 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 7, double, 1>;
template class InverseMassMatrixOperator<2, 7, double, 2>;
template class InverseMassMatrixOperator<2, 7, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_8 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 8, double, 1>;
template class InverseMassMatrixOperator<2, 8, double, 2>;
template class InverseMassMatrixOperator<2, 8, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_9 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 9, double, 1>;
template class InverseMassMatrixOperator<2, 9, double, 2>;
template class InverseMassMatrixOperator<2, 9, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_10 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 10, double, 1>;
template class InverseMassMatrixOperator<2, 10, double, 2>;
template class InverseMassMatrixOperator<2, 10, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_11 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 11, double, 1>;
template class InverseMassMatrixOperator<2, 11, double, 2>;
template class InverseMassMatrixOperator<2, 11, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_12 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 12, double, 1>;
template class InverseMassMatrixOperator<2, 12, double, 2>;
template class InverseMassMatrixOperator<2, 12, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_13 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 13, double, 1>;
template class InverseMassMatrixOperator<2, 13, double, 2>;
template class InverseMassMatrixOperator<2, 13, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_14 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 14, double, 1>;
template class InverseMassMatrixOperator<2, 14, double, 2>;
template class InverseMassMatrixOperator<2, 14, double, 2 + 2>;
#endif

#if DIM_2 && DEGREE_15 && OP_DOUBLE
template class InverseMassMatrixOperator<2, 15, double, 1>;
template class InverseMassMatrixOperator<2, 15, double, 2>;
template class InverseMassMatrixOperator<2, 15, double, 2 + 2>;
#endif
