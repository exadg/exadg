/*
 * operator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include <exadg/convection_diffusion/spatial_discretization/interface.h>

namespace ExaDG
{
namespace ConvDiff
{
namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure
// virtual.

// instantiations
template class Operator<float>;
template class Operator<double>;
} // namespace Interface

template class OperatorOIF<float>;
template class OperatorOIF<double>;

} // namespace ConvDiff
} // namespace ExaDG
