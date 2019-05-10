/*
 * dg_operator.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "dg_operator.h"

namespace CompNS
{
template class DGOperator<2, float>;
template class DGOperator<2, double>;

template class DGOperator<3, float>;
template class DGOperator<3, double>;

} // namespace CompNS
