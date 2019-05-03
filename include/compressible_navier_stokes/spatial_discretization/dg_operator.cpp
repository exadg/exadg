/*
 * dg_operator.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "dg_operator.h"

// instantiations
#include <navierstokes/config.h>

namespace CompNS
{
#if DIM_2 && OP_FLOAT
template class DGOperator<2, float>;
#endif
#if DIM_3 && OP_FLOAT
template class DGOperator<3, float>;
#endif

#if DIM_2 && OP_DOUBLE
template class DGOperator<2, double>;
#endif
#if DIM_3 && OP_DOUBLE
template class DGOperator<3, double>;
#endif

} // namespace CompNS
