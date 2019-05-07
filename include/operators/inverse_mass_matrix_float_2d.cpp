/*
 * inverse_mass_matrix.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "inverse_mass_matrix.h"

#include <navierstokes/config.h>

#if DIM_2 && DEGREE_0 && OP_FLOAT
template class InverseMassMatrixOperator<2, 0, float, 1>;
template class InverseMassMatrixOperator<2, 0, float, 2>;
template class InverseMassMatrixOperator<2, 0, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_1 && OP_FLOAT
template class InverseMassMatrixOperator<2, 1, float, 1>;
template class InverseMassMatrixOperator<2, 1, float, 2>;
template class InverseMassMatrixOperator<2, 1, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_2 && OP_FLOAT
template class InverseMassMatrixOperator<2, 2, float, 1>;
template class InverseMassMatrixOperator<2, 2, float, 2>;
template class InverseMassMatrixOperator<2, 2, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_3 && OP_FLOAT
template class InverseMassMatrixOperator<2, 3, float, 1>;
template class InverseMassMatrixOperator<2, 3, float, 2>;
template class InverseMassMatrixOperator<2, 3, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_4 && OP_FLOAT
template class InverseMassMatrixOperator<2, 4, float, 1>;
template class InverseMassMatrixOperator<2, 4, float, 2>;
template class InverseMassMatrixOperator<2, 4, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_5 && OP_FLOAT
template class InverseMassMatrixOperator<2, 5, float, 1>;
template class InverseMassMatrixOperator<2, 5, float, 2>;
template class InverseMassMatrixOperator<2, 5, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_6 && OP_FLOAT
template class InverseMassMatrixOperator<2, 6, float, 1>;
template class InverseMassMatrixOperator<2, 6, float, 2>;
template class InverseMassMatrixOperator<2, 6, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_7 && OP_FLOAT
template class InverseMassMatrixOperator<2, 7, float, 1>;
template class InverseMassMatrixOperator<2, 7, float, 2>;
template class InverseMassMatrixOperator<2, 7, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_8 && OP_FLOAT
template class InverseMassMatrixOperator<2, 8, float, 1>;
template class InverseMassMatrixOperator<2, 8, float, 2>;
template class InverseMassMatrixOperator<2, 8, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_9 && OP_FLOAT
template class InverseMassMatrixOperator<2, 9, float, 1>;
template class InverseMassMatrixOperator<2, 9, float, 2>;
template class InverseMassMatrixOperator<2, 9, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_10 && OP_FLOAT
template class InverseMassMatrixOperator<2, 10, float, 1>;
template class InverseMassMatrixOperator<2, 10, float, 2>;
template class InverseMassMatrixOperator<2, 10, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_11 && OP_FLOAT
template class InverseMassMatrixOperator<2, 11, float, 1>;
template class InverseMassMatrixOperator<2, 11, float, 2>;
template class InverseMassMatrixOperator<2, 11, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_12 && OP_FLOAT
template class InverseMassMatrixOperator<2, 12, float, 1>;
template class InverseMassMatrixOperator<2, 12, float, 2>;
template class InverseMassMatrixOperator<2, 12, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_13 && OP_FLOAT
template class InverseMassMatrixOperator<2, 13, float, 1>;
template class InverseMassMatrixOperator<2, 13, float, 2>;
template class InverseMassMatrixOperator<2, 13, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_14 && OP_FLOAT
template class InverseMassMatrixOperator<2, 14, float, 1>;
template class InverseMassMatrixOperator<2, 14, float, 2>;
template class InverseMassMatrixOperator<2, 14, float, 2 + 2>;
#endif

#if DIM_2 && DEGREE_15 && OP_FLOAT
template class InverseMassMatrixOperator<2, 15, float, 1>;
template class InverseMassMatrixOperator<2, 15, float, 2>;
template class InverseMassMatrixOperator<2, 15, float, 2 + 2>;
#endif
