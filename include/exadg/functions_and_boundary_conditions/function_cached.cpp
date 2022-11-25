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

// ExaDG
#include <exadg/functions_and_boundary_conditions/function_cached.h>

namespace ExaDG
{
template<int rank, int dim>
FunctionCached<rank, dim>::FunctionCached()
{
}

template<int rank, int dim>
void
FunctionCached<rank, dim>::set_data_pointer(
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>> const interface_data_)
{
  interface_data = interface_data_;
}

template class FunctionCached<0, 2>;
template class FunctionCached<0, 3>;
template class FunctionCached<1, 2>;
template class FunctionCached<1, 3>;

} // namespace ExaDG
