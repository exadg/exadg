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
template<int rank, int dim, typename number_type>
ContainerInterfaceData<rank, dim, number_type>::ContainerInterfaceData()
{
}

template<int rank, int dim, typename number_type>
typename ContainerInterfaceData<rank, dim, number_type>::data_type
ContainerInterfaceData<rank, dim, number_type>::get_data(unsigned int const q_index,
                                                         unsigned int const face,
                                                         unsigned int const q,
                                                         unsigned int const v) const
{
  Assert(map_vector_index.find(q_index) != map_vector_index.end(),
         dealii::ExcMessage("Specified q_index does not exist in map_vector_index."));

  Assert(map_solution.find(q_index) != map_solution.end(),
         dealii::ExcMessage("Specified q_index does not exist in map_solution."));

  Id                              id    = std::make_tuple(face, q, v);
  dealii::types::global_dof_index index = map_vector_index.find(q_index)->second.find(id)->second;

  ArraySolutionValues const & array_solution = map_solution.find(q_index)->second;
  Assert(index < array_solution.size(), dealii::ExcMessage("Index exceeds dimensions of vector."));

  return array_solution[index];
}

template<int rank, int dim, typename number_type>
std::vector<typename ContainerInterfaceData<rank, dim, number_type>::quad_index> const &
ContainerInterfaceData<rank, dim, number_type>::get_quad_indices()
{
  return quad_indices;
}

template<int rank, int dim, typename number_type>
typename ContainerInterfaceData<rank, dim, number_type>::ArrayQuadraturePoints &
ContainerInterfaceData<rank, dim, number_type>::get_array_q_points(quad_index const & q_index)
{
  return map_q_points[q_index];
}

template<int rank, int dim, typename number_type>
typename ContainerInterfaceData<rank, dim, number_type>::ArraySolutionValues &
ContainerInterfaceData<rank, dim, number_type>::get_array_solution(quad_index const & q_index)
{
  return map_solution[q_index];
}

template<int rank, int dim>
FunctionCached<rank, dim>::FunctionCached()
{
}

template<int rank, int dim>
typename FunctionCached<rank, dim>::data_type
FunctionCached<rank, dim>::tensor_value(unsigned int const face,
                                        unsigned int const q,
                                        unsigned int const v,
                                        unsigned int const quad_index) const
{
  return interface_data->get_data(quad_index, face, q, v);
}

template<int rank, int dim>
void
FunctionCached<rank, dim>::set_data_pointer(
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>> const interface_data_)
{
  interface_data = interface_data_;
}

template class ContainerInterfaceData<0, 2, double>;
template class ContainerInterfaceData<1, 2, double>;
template class ContainerInterfaceData<0, 3, double>;
template class ContainerInterfaceData<1, 3, double>;

template class FunctionCached<0, 2>;
template class FunctionCached<0, 3>;
template class FunctionCached<1, 2>;
template class FunctionCached<1, 3>;

} // namespace ExaDG
