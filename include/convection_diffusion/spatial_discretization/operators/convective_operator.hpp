#include "convective_operator.h"

#include <navierstokes/config.h>

namespace ConvDiff
{
#if DIM_2 && DEGREE_0 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 0, 0, float>;
#endif
#if DIM_2 && DEGREE_0 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 0, 0, double>;
#endif

#if DIM_2 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 0, 1, float>;
template class ConvectiveOperator<2, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 0, 1, double>;
template class ConvectiveOperator<2, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 1, 2, float>;
template class ConvectiveOperator<2, 2, 2, float>;
#endif
#if DIM_2 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 1, 2, double>;
template class ConvectiveOperator<2, 2, 2, double>;
#endif

#if DIM_2 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 2, 3, float>;
template class ConvectiveOperator<2, 3, 3, float>;
#endif
#if DIM_2 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 2, 3, double>;
template class ConvectiveOperator<2, 3, 3, double>;
#endif

#if DIM_2 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 3, 4, float>;
template class ConvectiveOperator<2, 4, 4, float>;
#endif
#if DIM_2 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 3, 4, double>;
template class ConvectiveOperator<2, 4, 4, double>;
#endif

#if DIM_2 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 4, 5, float>;
template class ConvectiveOperator<2, 5, 5, float>;
#endif
#if DIM_2 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 4, 5, double>;
template class ConvectiveOperator<2, 5, 5, double>;
#endif

#if DIM_2 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 5, 6, float>;
template class ConvectiveOperator<2, 6, 6, float>;
#endif
#if DIM_2 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 5, 6, double>;
template class ConvectiveOperator<2, 6, 6, double>;
#endif

#if DIM_2 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 6, 7, float>;
template class ConvectiveOperator<2, 7, 7, float>;
#endif
#if DIM_2 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 6, 7, double>;
template class ConvectiveOperator<2, 7, 7, double>;
#endif

#if DIM_2 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 7, 8, float>;
template class ConvectiveOperator<2, 8, 8, float>;
#endif
#if DIM_2 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 7, 8, double>;
template class ConvectiveOperator<2, 8, 8, double>;
#endif

#if DIM_2 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 8, 9, float>;
template class ConvectiveOperator<2, 9, 9, float>;
#endif
#if DIM_2 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 8, 9, double>;
template class ConvectiveOperator<2, 9, 9, double>;
#endif

#if DIM_2 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 9, 10, float>;
template class ConvectiveOperator<2, 10, 10, float>;
#endif
#if DIM_2 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 9, 10, double>;
template class ConvectiveOperator<2, 10, 10, double>;
#endif

#if DIM_2 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 10, 11, float>;
template class ConvectiveOperator<2, 11, 11, float>;
#endif
#if DIM_2 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 10, 11, double>;
template class ConvectiveOperator<2, 11, 11, double>;
#endif

#if DIM_2 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 11, 12, float>;
template class ConvectiveOperator<2, 12, 12, float>;
#endif
#if DIM_2 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 11, 12, double>;
template class ConvectiveOperator<2, 12, 12, double>;
#endif

#if DIM_2 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 12, 13, float>;
template class ConvectiveOperator<2, 13, 13, float>;
#endif
#if DIM_2 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 12, 13, double>;
template class ConvectiveOperator<2, 13, 13, double>;
#endif

#if DIM_2 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 13, 14, float>;
template class ConvectiveOperator<2, 14, 14, float>;
#endif
#if DIM_2 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 13, 14, double>;
template class ConvectiveOperator<2, 14, 14, double>;
#endif

#if DIM_2 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 14, 15, float>;
template class ConvectiveOperator<2, 15, 15, float>;
#endif
#if DIM_2 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 14, 15, double>;
template class ConvectiveOperator<2, 15, 15, double>;
#endif

#if DIM_3 && DEGREE_0 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 0, 0, float>;
#endif
#if DIM_3 && DEGREE_0 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 0, 0, double>;
#endif

#if DIM_3 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 0, 1, float>;
template class ConvectiveOperator<3, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 0, 1, double>;
template class ConvectiveOperator<3, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 1, 2, float>;
template class ConvectiveOperator<3, 2, 2, float>;
#endif
#if DIM_3 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 1, 2, double>;
template class ConvectiveOperator<3, 2, 2, double>;
#endif

#if DIM_3 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 2, 3, float>;
template class ConvectiveOperator<3, 3, 3, float>;
#endif
#if DIM_3 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 2, 3, double>;
template class ConvectiveOperator<3, 3, 3, double>;
#endif

#if DIM_3 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 3, 4, float>;
template class ConvectiveOperator<3, 4, 4, float>;
#endif
#if DIM_3 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 3, 4, double>;
template class ConvectiveOperator<3, 4, 4, double>;
#endif

#if DIM_3 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 4, 5, float>;
template class ConvectiveOperator<3, 5, 5, float>;
#endif
#if DIM_3 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 4, 5, double>;
template class ConvectiveOperator<3, 5, 5, double>;
#endif

#if DIM_3 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 5, 6, float>;
template class ConvectiveOperator<3, 6, 6, float>;
#endif
#if DIM_3 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 5, 6, double>;
template class ConvectiveOperator<3, 6, 6, double>;
#endif

#if DIM_3 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 6, 7, float>;
template class ConvectiveOperator<3, 7, 7, float>;
#endif
#if DIM_3 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 6, 7, double>;
template class ConvectiveOperator<3, 7, 7, double>;
#endif

#if DIM_3 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 7, 8, float>;
template class ConvectiveOperator<3, 8, 8, float>;
#endif
#if DIM_3 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 7, 8, double>;
template class ConvectiveOperator<3, 8, 8, double>;
#endif

#if DIM_3 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 8, 9, float>;
template class ConvectiveOperator<3, 9, 9, float>;
#endif
#if DIM_3 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 8, 9, double>;
template class ConvectiveOperator<3, 9, 9, double>;
#endif

#if DIM_3 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 9, 10, float>;
template class ConvectiveOperator<3, 10, 10, float>;
#endif
#if DIM_3 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 9, 10, double>;
template class ConvectiveOperator<3, 10, 10, double>;
#endif

#if DIM_3 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 10, 11, float>;
template class ConvectiveOperator<3, 11, 11, float>;
#endif
#if DIM_3 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 10, 11, double>;
template class ConvectiveOperator<3, 11, 11, double>;
#endif

#if DIM_3 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 11, 12, float>;
template class ConvectiveOperator<3, 12, 12, float>;
#endif
#if DIM_3 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 11, 12, double>;
template class ConvectiveOperator<3, 12, 12, double>;
#endif

#if DIM_3 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 12, 13, float>;
template class ConvectiveOperator<3, 13, 13, float>;
#endif
#if DIM_3 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 12, 13, double>;
template class ConvectiveOperator<3, 13, 13, double>;
#endif

#if DIM_3 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 13, 14, float>;
template class ConvectiveOperator<3, 14, 14, float>;
#endif
#if DIM_3 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 13, 14, double>;
template class ConvectiveOperator<3, 14, 14, double>;
#endif

#if DIM_3 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 14, 15, float>;
template class ConvectiveOperator<3, 15, 15, float>;
#endif
#if DIM_3 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 14, 15, double>;
template class ConvectiveOperator<3, 15, 15, double>;
#endif

} // namespace ConvDiff
