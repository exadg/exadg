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
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>

namespace ExaDG
{
template<int rank, int dim>
ContainerInterfaceData<rank, dim>::ContainerInterfaceData()
{
}

template<int rank, int dim>
typename ContainerInterfaceData<rank, dim>::value_type
ContainerInterfaceData<rank, dim>::get_data(unsigned int const q_index,
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

template<int rank, int dim>
std::vector<typename ContainerInterfaceData<rank, dim>::quad_index> const &
ContainerInterfaceData<rank, dim>::get_quad_indices()
{
  return quad_indices;
}

template<int rank, int dim>
typename ContainerInterfaceData<rank, dim>::ArrayQuadraturePoints &
ContainerInterfaceData<rank, dim>::get_array_q_points(quad_index const & q_index)
{
  return map_q_points[q_index];
}

template<int rank, int dim>
typename ContainerInterfaceData<rank, dim>::ArraySolutionValues &
ContainerInterfaceData<rank, dim>::get_array_solution(quad_index const & q_index)
{
  return map_solution[q_index];
}

template<int rank, int dim, typename Number>
InterfaceCoupling<rank, dim, Number>::InterfaceCoupling() : dof_handler_src(nullptr)
{
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::setup(
  std::shared_ptr<ContainerInterfaceData<rank, dim>> interface_data_dst_,
  dealii::DoFHandler<dim> const &                    dof_handler_src_,
  dealii::Mapping<dim> const &                       mapping_src_,
  std::vector<bool> const &                          marked_vertices_src_,
  double const                                       tolerance_)
{
  AssertThrow(interface_data_dst_.get(),
              dealii::ExcMessage("Received uninitialized variable. Aborting."));

  if(marked_vertices_src_.size() > 0)
  {
    AssertThrow(marked_vertices_src_.size() ==
                  (unsigned int)dof_handler_src_.get_triangulation().n_vertices(),
                dealii::ExcMessage("Vector marked_vertices_src_ has invalid size."));
  }

  interface_data_dst = interface_data_dst_;
  dof_handler_src    = &dof_handler_src_;

  for(auto quad_index : interface_data_dst->get_quad_indices())
  {
    // exchange quadrature points with their owners
    map_evaluator.emplace(quad_index,
                          dealii::Utilities::MPI::RemotePointEvaluation<dim>(
                            tolerance_, false, 0, [marked_vertices_src_]() {
                              return marked_vertices_src_;
                            }));

    map_evaluator[quad_index].reinit(interface_data_dst->get_array_q_points(quad_index),
                                     dof_handler_src_.get_triangulation(),
                                     mapping_src_);

    AssertThrow(
      map_evaluator[quad_index].all_points_found() == true,
      dealii::ExcMessage(
        "Setup of InterfaceCoupling was not successful. Not all points have been found."));
  }
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::update_data(VectorType const & dof_vector_src)
{
  dof_vector_src.update_ghost_values();

  for(auto quadrature : interface_data_dst->get_quad_indices())
  {
    auto const result =
      dealii::VectorTools::point_values<n_components>(map_evaluator[quadrature],
                                                      *dof_handler_src,
                                                      dof_vector_src,
                                                      dealii::VectorTools::EvaluationFlags::avg);

    auto & array_solution = interface_data_dst->get_array_solution(quadrature);

    Assert(result.size() == array_solution.size(),
           dealii::ExcMessage("Vectors must have the same length."));

    for(unsigned int i = 0; i < result.size(); ++i)
      array_solution[i] = result[i];
  }
}

template class ContainerInterfaceData<0, 2>;
template class ContainerInterfaceData<1, 2>;
template class ContainerInterfaceData<0, 3>;
template class ContainerInterfaceData<1, 3>;

template class InterfaceCoupling<0, 2, float>;
template class InterfaceCoupling<1, 2, float>;
template class InterfaceCoupling<0, 3, float>;
template class InterfaceCoupling<1, 3, float>;

template class InterfaceCoupling<0, 2, double>;
template class InterfaceCoupling<1, 2, double>;
template class InterfaceCoupling<0, 3, double>;
template class InterfaceCoupling<1, 3, double>;

} // namespace ExaDG
