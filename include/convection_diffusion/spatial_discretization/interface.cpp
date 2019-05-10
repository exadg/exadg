/*
 * operator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "interface.h"

namespace ConvDiff
{
namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations
template class Operator<float>;
template class OperatorOIF<float>;

template class Operator<double>;
template class OperatorOIF<double>;

} // namespace Interface

} // namespace ConvDiff
