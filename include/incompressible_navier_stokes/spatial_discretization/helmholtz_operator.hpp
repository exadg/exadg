#include "helmholtz_operator.h"

#include <navierstokes/config.h>

namespace IncNS
{
#if DIM_2 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 1, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 1, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 2, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 2, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 3, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 3, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 4, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 4, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 5, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 5, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 6, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 6, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 7, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 7, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 8, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 8, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 9, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 9, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 10, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 10, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 11, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 11, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 12, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 12, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 13, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 13, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 14, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 14, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<2, 15, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<2, 15, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 1, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 1, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 2, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 2, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 3, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 3, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 4, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 4, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 5, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 5, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 6, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 6, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 7, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 7, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 8, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 8, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 9, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 9, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_10 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 10, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_10 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 10, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_11 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 11, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_11 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 11, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_12 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 12, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_12 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 12, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_13 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 13, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_13 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 13, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_14 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 14, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_14 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 14, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_15 && (OP_FLOAT || MG_FLOAT)
template class HelmholtzOperator<3, 15, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_15 && (OP_DOUBLE || MG_DOUBLE)
template class HelmholtzOperator<3, 15, 1, 1, double>;
#endif

} // namespace IncNS
