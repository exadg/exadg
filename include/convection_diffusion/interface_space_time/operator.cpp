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
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations
#include <navierstokes/config.h>

#if OP_FLOAT
template class Operator<float>;
template class OperatorOIF<float>;
#endif

#if OP_DOUBLE
template class Operator<double>;
template class OperatorOIF<double>;
#endif

} // namespace Interface

} // namespace ConvDiff
