/*
 * body_force_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "body_force_operator.h"

namespace IncNS
{
// currently implemented in header-file

template class BodyForceOperator<2, float>;
template class BodyForceOperator<2, double>;

template class BodyForceOperator<3, float>;
template class BodyForceOperator<3, double>;

} // namespace IncNS
