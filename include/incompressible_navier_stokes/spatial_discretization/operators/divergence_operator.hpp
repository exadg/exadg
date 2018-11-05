/*
 * divergence_operator.hpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include <navierstokes/config.h>

namespace IncNS
{
#if DIM_2 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 1, 0, float>;
template class DivergenceOperator<2, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 1, 0, double>;
template class DivergenceOperator<2, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 2, 1, float>;
template class DivergenceOperator<2, 2, 2, float>;
#endif
#if DIM_2 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 2, 1, double>;
template class DivergenceOperator<2, 2, 2, double>;
#endif

#if DIM_2 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 3, 2, float>;
template class DivergenceOperator<2, 3, 3, float>;
#endif
#if DIM_2 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 3, 2, double>;
template class DivergenceOperator<2, 3, 3, double>;
#endif

#if DIM_2 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 4, 3, float>;
template class DivergenceOperator<2, 4, 4, float>;
#endif
#if DIM_2 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 4, 3, double>;
template class DivergenceOperator<2, 4, 4, double>;
#endif

#if DIM_2 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 5, 4, float>;
template class DivergenceOperator<2, 5, 5, float>;
#endif
#if DIM_2 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 5, 4, double>;
template class DivergenceOperator<2, 5, 5, double>;
#endif

#if DIM_2 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 6, 5, float>;
template class DivergenceOperator<2, 6, 6, float>;
#endif
#if DIM_2 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 6, 5, double>;
template class DivergenceOperator<2, 6, 6, double>;
#endif

#if DIM_2 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 7, 6, float>;
template class DivergenceOperator<2, 7, 7, float>;
#endif
#if DIM_2 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 7, 6, double>;
template class DivergenceOperator<2, 7, 7, double>;
#endif

#if DIM_2 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 8, 7, float>;
template class DivergenceOperator<2, 8, 8, float>;
#endif
#if DIM_2 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 8, 7, double>;
template class DivergenceOperator<2, 8, 8, double>;
#endif

#if DIM_2 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 9, 8, float>;
template class DivergenceOperator<2, 9, 9, float>;
#endif
#if DIM_2 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 9, 8, double>;
template class DivergenceOperator<2, 9, 9, double>;
#endif

#if DIM_2 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 10, 9, float>;
template class DivergenceOperator<2, 10, 10, float>;
#endif
#if DIM_2 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 10, 9, double>;
template class DivergenceOperator<2, 10, 10, double>;
#endif

#if DIM_2 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 11, 10, float>;
template class DivergenceOperator<2, 11, 11, float>;
#endif
#if DIM_2 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 11, 10, double>;
template class DivergenceOperator<2, 11, 11, double>;
#endif

#if DIM_2 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 12, 11, float>;
template class DivergenceOperator<2, 12, 12, float>;
#endif
#if DIM_2 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 12, 11, double>;
template class DivergenceOperator<2, 12, 12, double>;
#endif

#if DIM_2 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 13, 12, float>;
template class DivergenceOperator<2, 13, 13, float>;
#endif
#if DIM_2 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 13, 12, double>;
template class DivergenceOperator<2, 13, 13, double>;
#endif

#if DIM_2 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 14, 13, float>;
template class DivergenceOperator<2, 14, 14, float>;
#endif
#if DIM_2 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 14, 13, double>;
template class DivergenceOperator<2, 14, 14, double>;
#endif

#if DIM_2 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<2, 15, 14, float>;
template class DivergenceOperator<2, 15, 15, float>;
#endif
#if DIM_2 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<2, 15, 14, double>;
template class DivergenceOperator<2, 15, 15, double>;
#endif


#if DIM_3 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 1, 0, float>;
template class DivergenceOperator<3, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 1, 0, double>;
template class DivergenceOperator<3, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 2, 1, float>;
template class DivergenceOperator<3, 2, 2, float>;
#endif
#if DIM_3 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 2, 1, double>;
template class DivergenceOperator<3, 2, 2, double>;
#endif

#if DIM_3 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 3, 2, float>;
template class DivergenceOperator<3, 3, 3, float>;
#endif
#if DIM_3 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 3, 2, double>;
template class DivergenceOperator<3, 3, 3, double>;
#endif

#if DIM_3 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 4, 3, float>;
template class DivergenceOperator<3, 4, 4, float>;
#endif
#if DIM_3 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 4, 3, double>;
template class DivergenceOperator<3, 4, 4, double>;
#endif

#if DIM_3 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 5, 4, float>;
template class DivergenceOperator<3, 5, 5, float>;
#endif
#if DIM_3 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 5, 4, double>;
template class DivergenceOperator<3, 5, 5, double>;
#endif

#if DIM_3 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 6, 5, float>;
template class DivergenceOperator<3, 6, 6, float>;
#endif
#if DIM_3 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 6, 5, double>;
template class DivergenceOperator<3, 6, 6, double>;
#endif

#if DIM_3 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 7, 6, float>;
template class DivergenceOperator<3, 7, 7, float>;
#endif
#if DIM_3 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 7, 6, double>;
template class DivergenceOperator<3, 7, 7, double>;
#endif

#if DIM_3 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 8, 7, float>;
template class DivergenceOperator<3, 8, 8, float>;
#endif
#if DIM_3 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 8, 7, double>;
template class DivergenceOperator<3, 8, 8, double>;
#endif

#if DIM_3 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 9, 8, float>;
template class DivergenceOperator<3, 9, 9, float>;
#endif
#if DIM_3 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 9, 8, double>;
template class DivergenceOperator<3, 9, 9, double>;
#endif

#if DIM_3 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 10, 9, float>;
template class DivergenceOperator<3, 10, 10, float>;
#endif
#if DIM_3 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 10, 9, double>;
template class DivergenceOperator<3, 10, 10, double>;
#endif

#if DIM_3 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 11, 10, float>;
template class DivergenceOperator<3, 11, 11, float>;
#endif
#if DIM_3 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 11, 10, double>;
template class DivergenceOperator<3, 11, 11, double>;
#endif

#if DIM_3 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 12, 11, float>;
template class DivergenceOperator<3, 12, 12, float>;
#endif
#if DIM_3 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 12, 11, double>;
template class DivergenceOperator<3, 12, 12, double>;
#endif

#if DIM_3 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 13, 12, float>;
template class DivergenceOperator<3, 13, 13, float>;
#endif
#if DIM_3 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 13, 12, double>;
template class DivergenceOperator<3, 13, 13, double>;
#endif

#if DIM_3 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 14, 13, float>;
template class DivergenceOperator<3, 14, 14, float>;
#endif
#if DIM_3 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 14, 13, double>;
template class DivergenceOperator<3, 14, 14, double>;
#endif

#if DIM_3 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class DivergenceOperator<3, 15, 14, float>;
template class DivergenceOperator<3, 15, 15, float>;
#endif
#if DIM_3 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class DivergenceOperator<3, 15, 14, double>;
template class DivergenceOperator<3, 15, 15, double>;
#endif
} // namespace IncNS
