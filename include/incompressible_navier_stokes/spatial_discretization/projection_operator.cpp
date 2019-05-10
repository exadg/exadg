/*
 * projection_operator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "projection_operator.h"

namespace IncNS
{
// all functions are currently implemented in .h-file.

template class ProjectionOperator<2, float>;
template class ProjectionOperator<2, double>;

template class ProjectionOperator<3, float>;
template class ProjectionOperator<3, double>;

} // namespace IncNS
