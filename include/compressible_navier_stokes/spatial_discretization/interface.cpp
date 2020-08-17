/*
 * operator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "interface.h"

namespace ExaDG
{
namespace CompNS
{
namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations

template class Operator<float>;
template class Operator<double>;

} // namespace Interface
} // namespace CompNS
} // namespace ExaDG
