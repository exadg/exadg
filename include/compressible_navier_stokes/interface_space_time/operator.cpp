/*
 * operator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "operator.h"

namespace CompNS
{
namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations
#include <navierstokes/config.h>

#if OP_FLOAT
template class Operator<float>;
#endif
#if OP_DOUBLE
template class Operator<double>;
#endif

} // namespace Interface

} // namespace CompNS
