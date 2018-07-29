#include "helmholtz_operator.h"

#include <navier-constants.h>

namespace IncNS
{

#if DIM_2 && DEGREE_1
template class HelmholtzOperator<2, 1, 1, 1, float>;
template class HelmholtzOperator<2, 1, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_2
template class HelmholtzOperator<2, 2, 1, 1, float>;
template class HelmholtzOperator<2, 2, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_3
template class HelmholtzOperator<2, 3, 1, 1, float>;
template class HelmholtzOperator<2, 3, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_4
template class HelmholtzOperator<2, 4, 1, 1, float>;
template class HelmholtzOperator<2, 4, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_5
template class HelmholtzOperator<2, 5, 1, 1, float>;
template class HelmholtzOperator<2, 5, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_6
template class HelmholtzOperator<2, 6, 1, 1, float>;
template class HelmholtzOperator<2, 6, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_7
template class HelmholtzOperator<2, 7, 1, 1, float>;
template class HelmholtzOperator<2, 7, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_8
template class HelmholtzOperator<2, 8, 1, 1, float>;
template class HelmholtzOperator<2, 8, 1, 1, double>;
#endif

#if DIM_2 && DEGREE_9
template class HelmholtzOperator<2, 9, 1, 1, float>;
template class HelmholtzOperator<2, 9, 1, 1, double>;
#endif

}

