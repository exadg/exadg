/*
 * interface.cpp
 *
 *  Created on: 02.05.2020
 *      Author: fehn
 */

#include <exadg/structure/spatial_discretization/interface.h>

namespace ExaDG
{
namespace Structure
{
namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations

template class Operator<float>;
template class Operator<double>;

} // namespace Interface
} // namespace Structure
} // namespace ExaDG
