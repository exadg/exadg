/*
 * inverse_mass_matrix.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "inverse_mass_matrix.h"

#include <navierstokes/config.h>

#if DIM_3 && DEGREE_0 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 0, double, 1>;
template class InverseMassMatrixOperator<3, 0, double, 3>;
template class InverseMassMatrixOperator<3, 0, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_1 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 1, double, 1>;
template class InverseMassMatrixOperator<3, 1, double, 3>;
template class InverseMassMatrixOperator<3, 1, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_2 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 2, double, 1>;
template class InverseMassMatrixOperator<3, 2, double, 3>;
template class InverseMassMatrixOperator<3, 2, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_3 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 3, double, 1>;
template class InverseMassMatrixOperator<3, 3, double, 3>;
template class InverseMassMatrixOperator<3, 3, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_4 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 4, double, 1>;
template class InverseMassMatrixOperator<3, 4, double, 3>;
template class InverseMassMatrixOperator<3, 4, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_5 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 5, double, 1>;
template class InverseMassMatrixOperator<3, 5, double, 3>;
template class InverseMassMatrixOperator<3, 5, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_6 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 6, double, 1>;
template class InverseMassMatrixOperator<3, 6, double, 3>;
template class InverseMassMatrixOperator<3, 6, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_7 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 7, double, 1>;
template class InverseMassMatrixOperator<3, 7, double, 3>;
template class InverseMassMatrixOperator<3, 7, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_8 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 8, double, 1>;
template class InverseMassMatrixOperator<3, 8, double, 3>;
template class InverseMassMatrixOperator<3, 8, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_9 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 9, double, 1>;
template class InverseMassMatrixOperator<3, 9, double, 3>;
template class InverseMassMatrixOperator<3, 9, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_10 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 10, double, 1>;
template class InverseMassMatrixOperator<3, 10, double, 3>;
template class InverseMassMatrixOperator<3, 10, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_11 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 11, double, 1>;
template class InverseMassMatrixOperator<3, 11, double, 3>;
template class InverseMassMatrixOperator<3, 11, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_12 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 12, double, 1>;
template class InverseMassMatrixOperator<3, 12, double, 3>;
template class InverseMassMatrixOperator<3, 12, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_13 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 13, double, 1>;
template class InverseMassMatrixOperator<3, 13, double, 3>;
template class InverseMassMatrixOperator<3, 13, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_14 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 14, double, 1>;
template class InverseMassMatrixOperator<3, 14, double, 3>;
template class InverseMassMatrixOperator<3, 14, double, 3 + 2>;
#endif

#if DIM_3 && DEGREE_15 && OP_DOUBLE
template class InverseMassMatrixOperator<3, 15, double, 1>;
template class InverseMassMatrixOperator<3, 15, double, 3>;
template class InverseMassMatrixOperator<3, 15, double, 3 + 2>;
#endif
