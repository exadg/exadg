/*
 * operator.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include "interface.h"

namespace IncNS
{
namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations

// float
template class OperatorBase<2, float>;
template class OperatorBase<3, float>;
template class OperatorCoupled<float>;
template class OperatorDualSplitting<float>;
template class OperatorPressureCorrection<float>;
template class OperatorOIF<2, float>;
template class OperatorOIF<3, float>;

// double
template class OperatorBase<2, double>;
template class OperatorBase<3, double>;
template class OperatorCoupled<double>;
template class OperatorDualSplitting<double>;
template class OperatorPressureCorrection<double>;
template class OperatorOIF<2, double>;
template class OperatorOIF<3, double>;

} // namespace Interface

} // namespace IncNS
