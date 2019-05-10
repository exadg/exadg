/*
 * gradient_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "gradient_operator.h"

namespace IncNS
{
// currently implemented in header-file

template class GradientOperator<2, float>;
template class GradientOperator<2, double>;

template class GradientOperator<3, float>;
template class GradientOperator<3, double>;

} // namespace IncNS
