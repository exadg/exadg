/*
 * operator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "operator.h"

namespace ConvDiff
{

namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure virtual.

// instantiate for float and double
template class Operator<float>;
template class Operator<double>;

template class OperatorOIF<float>;
template class OperatorOIF<double>;

} // namespace Interface

} // namespace ConvDiff
