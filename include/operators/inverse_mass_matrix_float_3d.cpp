/*
 * inverse_mass_matrix.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "inverse_mass_matrix.h"

#include <navierstokes/config.h>

#if DIM_3 && DEGREE_0 && OP_FLOAT
template class InverseMassMatrixOperator<3, 0, float, 1>;
template class InverseMassMatrixOperator<3, 0, float, 3>;
template class InverseMassMatrixOperator<3, 0, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_1 && OP_FLOAT
template class InverseMassMatrixOperator<3, 1, float, 1>;
template class InverseMassMatrixOperator<3, 1, float, 3>;
template class InverseMassMatrixOperator<3, 1, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_2 && OP_FLOAT
template class InverseMassMatrixOperator<3, 2, float, 1>;
template class InverseMassMatrixOperator<3, 2, float, 3>;
template class InverseMassMatrixOperator<3, 2, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_3 && OP_FLOAT
template class InverseMassMatrixOperator<3, 3, float, 1>;
template class InverseMassMatrixOperator<3, 3, float, 3>;
template class InverseMassMatrixOperator<3, 3, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_4 && OP_FLOAT
template class InverseMassMatrixOperator<3, 4, float, 1>;
template class InverseMassMatrixOperator<3, 4, float, 3>;
template class InverseMassMatrixOperator<3, 4, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_5 && OP_FLOAT
template class InverseMassMatrixOperator<3, 5, float, 1>;
template class InverseMassMatrixOperator<3, 5, float, 3>;
template class InverseMassMatrixOperator<3, 5, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_6 && OP_FLOAT
template class InverseMassMatrixOperator<3, 6, float, 1>;
template class InverseMassMatrixOperator<3, 6, float, 3>;
template class InverseMassMatrixOperator<3, 6, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_7 && OP_FLOAT
template class InverseMassMatrixOperator<3, 7, float, 1>;
template class InverseMassMatrixOperator<3, 7, float, 3>;
template class InverseMassMatrixOperator<3, 7, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_8 && OP_FLOAT
template class InverseMassMatrixOperator<3, 8, float, 1>;
template class InverseMassMatrixOperator<3, 8, float, 3>;
template class InverseMassMatrixOperator<3, 8, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_9 && OP_FLOAT
template class InverseMassMatrixOperator<3, 9, float, 1>;
template class InverseMassMatrixOperator<3, 9, float, 3>;
template class InverseMassMatrixOperator<3, 9, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_10 && OP_FLOAT
template class InverseMassMatrixOperator<3, 10, float, 1>;
template class InverseMassMatrixOperator<3, 10, float, 3>;
template class InverseMassMatrixOperator<3, 10, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_11 && OP_FLOAT
template class InverseMassMatrixOperator<3, 11, float, 1>;
template class InverseMassMatrixOperator<3, 11, float, 3>;
template class InverseMassMatrixOperator<3, 11, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_12 && OP_FLOAT
template class InverseMassMatrixOperator<3, 12, float, 1>;
template class InverseMassMatrixOperator<3, 12, float, 3>;
template class InverseMassMatrixOperator<3, 12, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_13 && OP_FLOAT
template class InverseMassMatrixOperator<3, 13, float, 1>;
template class InverseMassMatrixOperator<3, 13, float, 3>;
template class InverseMassMatrixOperator<3, 13, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_14 && OP_FLOAT
template class InverseMassMatrixOperator<3, 14, float, 1>;
template class InverseMassMatrixOperator<3, 14, float, 3>;
template class InverseMassMatrixOperator<3, 14, float, 3 + 2>;
#endif

#if DIM_3 && DEGREE_15 && OP_FLOAT
template class InverseMassMatrixOperator<3, 15, float, 1>;
template class InverseMassMatrixOperator<3, 15, float, 3>;
template class InverseMassMatrixOperator<3, 15, float, 3 + 2>;
#endif
