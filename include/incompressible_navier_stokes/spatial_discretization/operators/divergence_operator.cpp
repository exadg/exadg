/*
 * divergence_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "divergence_operator.h"

namespace IncNS
{
// currently implemented in header-file

template class DivergenceOperator<2, float>;
template class DivergenceOperator<2, double>;

template class DivergenceOperator<3, float>;
template class DivergenceOperator<3, double>;

} // namespace IncNS
