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
template class OperatorBase<float>;
template class OperatorCoupled<float>;
template class OperatorDualSplitting<float>;
template class OperatorPressureCorrection<float>;
template class OperatorOIF<float>;

// double
template class OperatorBase<double>;
template class OperatorCoupled<double>;
template class OperatorDualSplitting<double>;
template class OperatorPressureCorrection<double>;
template class OperatorOIF<double>;

} // namespace Interface

} // namespace IncNS
