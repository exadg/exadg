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

template class OperatorOIF<float>;
template class OperatorOIF<double>;

} // namespace Interface

} // namespace IncNS
