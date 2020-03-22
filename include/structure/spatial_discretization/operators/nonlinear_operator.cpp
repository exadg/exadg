/*
 * nonlinear_operator.cpp
 *
 *  Created on: 18.03.2020
 *      Author: fehn
 */

#include "nonlinear_operator.h"

namespace Structure
{
template class NonLinearOperator<2, float>;
template class NonLinearOperator<2, double>;

template class NonLinearOperator<3, float>;
template class NonLinearOperator<3, double>;

} // namespace Structure
