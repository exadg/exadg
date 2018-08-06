#include "velocity_convection_diffusion_operator.h"

#include <navier_constants.h>

namespace IncNS
{

#if DIM_2 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 1, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 1, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 2, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 2, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 3, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 3, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 4, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 4, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 5, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 5, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 6, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 6, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 7, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 7, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 8, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 8, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<2, 9, 1, 1, float>;
#endif
#if DIM_2 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<2, 9, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 1, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 1, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 2, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 2, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 3, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 3, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 4, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 4, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 5, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 5, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 6, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 6, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 7, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 7, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 8, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 8, 1, 1, double>;
#endif

#if DIM_3 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class VelocityConvDiffOperator<3, 9, 1, 1, float>;
#endif
#if DIM_3 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class VelocityConvDiffOperator<3, 9, 1, 1, double>;
#endif

}

