/*
 * body_force_operator.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "body_force_operator.h"


namespace Structure
{
template class BodyForceOperator<2, float>;
template class BodyForceOperator<2, double>;

template class BodyForceOperator<3, float>;
template class BodyForceOperator<3, double>;

} // namespace Structure
