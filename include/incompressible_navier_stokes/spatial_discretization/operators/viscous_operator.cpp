/*
 * viscous_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */


#include "viscous_operator.h"

namespace IncNS
{
// currently implemented in header-file

template class ViscousOperator<2, float>;
template class ViscousOperator<2, double>;

template class ViscousOperator<3, float>;
template class ViscousOperator<3, double>;

} // namespace IncNS
