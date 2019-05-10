/*
 * mass_matrix_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "mass_matrix_operator.h"

namespace IncNS
{
// currently implemented in header-file

template class MassMatrixOperator<2, float>;
template class MassMatrixOperator<2, double>;

template class MassMatrixOperator<3, float>;
template class MassMatrixOperator<3, double>;

} // namespace IncNS
