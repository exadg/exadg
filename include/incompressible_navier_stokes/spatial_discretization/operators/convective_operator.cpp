/*
 * convective_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "convective_operator.h"

namespace IncNS
{
// currently implemented in header-file

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace IncNS
