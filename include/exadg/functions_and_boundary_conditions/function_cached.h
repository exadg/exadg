/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_
#define INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_

// deal.II
#include <deal.II/base/tensor.h>

#include <map>
#include <tuple>
#include <vector>

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>

namespace ExaDG
{
/*
 * The only reason why we do not integrate ContainerInterfaceData directly into
 * FunctionCached is that we want to use only one object of type
 * ContainerInterfaceData for (potentially) many boundary IDs and, therefore,
 * many objects of type FunctionCached.
 */
template<int rank, int dim>
class FunctionCached
{
private:
  typedef typename ContainerInterfaceData<rank, dim>::value_type value_type;

public:
  FunctionCached();

  // read data
  value_type
  tensor_value(unsigned int const face,
               unsigned int const q,
               unsigned int const v,
               unsigned int const quad_index) const;

  // initialize data pointer
  void
  set_data_pointer(std::shared_ptr<ContainerInterfaceData<rank, dim>> const interface_data_);

private:
  std::shared_ptr<ContainerInterfaceData<rank, dim>> interface_data;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_ */
