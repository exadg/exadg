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
template<int dim, int n_components, typename Number>
ContainerInterfaceData<dim, n_components, Number>::ContainerInterfaceData()
{
}

template<int dim, int n_components, typename Number>
void
ContainerInterfaceData<dim, n_components, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_,
  unsigned int const                               dof_index_,
  std::vector<quad_index> const &                  quad_indices_,
  MapBoundaryCondition const &                     map_bc_)
{
  quad_indices = quad_indices_;

  for(auto q_index : quad_indices)
  {
    // initialize maps
    map_vector_index.emplace(q_index, MapVectorIndex());
    map_q_points.emplace(q_index, ArrayQuadraturePoints());
    map_solution.emplace(q_index, ArraySolutionValues());

    MapVectorIndex &        map_index          = map_vector_index.find(q_index)->second;
    ArrayQuadraturePoints & array_q_points_dst = map_q_points.find(q_index)->second;
    ArraySolutionValues &   array_solution_dst = map_solution.find(q_index)->second;

    // create map "ID = {face, q, v} <-> vector_index" and fill array of quadrature points
    for(unsigned int face = matrix_free_->n_inner_face_batches();
        face < matrix_free_->n_inner_face_batches() + matrix_free_->n_boundary_face_batches();
        ++face)
    {
      // only consider relevant boundary IDs
      if(map_bc_.find(matrix_free_->get_boundary_id(face)) != map_bc_.end())
      {
        FaceIntegrator<dim, n_components, Number> integrator(*matrix_free_,
                                                             true,
                                                             dof_index_,
                                                             q_index);
        integrator.reinit(face);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          dealii::Point<dim, dealii::VectorizedArray<Number>> q_points =
            integrator.quadrature_point(q);

          for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
          {
            dealii::Point<dim> q_point;
            for(unsigned int d = 0; d < dim; ++d)
              q_point[d] = q_points[d][v];

            Id                              id    = std::make_tuple(face, q, v);
            dealii::types::global_dof_index index = array_q_points_dst.size();
            map_index.emplace(id, index);
            array_q_points_dst.push_back(q_point);
          }
        }
      }
    }

    array_solution_dst.resize(array_q_points_dst.size(), value_type());
  }

  // finally, give boundary condition access to the data
  for(auto boundary : map_bc_)
  {
    boundary.second->set_data_pointer(map_vector_index, map_solution);
  }
}

template<int dim, int n_components, typename Number>
std::vector<typename ContainerInterfaceData<dim, n_components, Number>::quad_index> const &
ContainerInterfaceData<dim, n_components, Number>::get_quad_indices()
{
  return quad_indices;
}

template<int dim, int n_components, typename Number>
typename ContainerInterfaceData<dim, n_components, Number>::ArrayQuadraturePoints &
ContainerInterfaceData<dim, n_components, Number>::get_array_q_points(quad_index const & q_index)
{
  return map_q_points[q_index];
}

template<int dim, int n_components, typename Number>
typename ContainerInterfaceData<dim, n_components, Number>::ArraySolutionValues &
ContainerInterfaceData<dim, n_components, Number>::get_array_solution(quad_index const & q_index)
{
  return map_solution[q_index];
}

template<int dim, int n_components, typename Number>
InterfaceCoupling<dim, n_components, Number>::InterfaceCoupling() : dof_handler_src(nullptr)
{
}

template<int dim, int n_components, typename Number>
void
InterfaceCoupling<dim, n_components, Number>::setup(
  std::shared_ptr<ContainerInterfaceData<dim, n_components, Number>> interface_data_dst_,
  dealii::DoFHandler<dim> const &                                    dof_handler_src_,
  dealii::Mapping<dim> const &                                       mapping_src_,
  std::vector<bool> const &                                          marked_vertices_src_,
  double const                                                       tolerance_)
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

template<int dim, int n_components, typename Number>
void
InterfaceCoupling<dim, n_components, Number>::update_data(VectorType const & dof_vector_src)
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

template class ContainerInterfaceData<2, 1, float>;
template class ContainerInterfaceData<2, 2, float>;
template class ContainerInterfaceData<3, 1, float>;
template class ContainerInterfaceData<3, 3, float>;

template class ContainerInterfaceData<2, 1, double>;
template class ContainerInterfaceData<2, 2, double>;
template class ContainerInterfaceData<3, 1, double>;
template class ContainerInterfaceData<3, 3, double>;

// template class InterfaceCoupling<2, 1, float>;
// template class InterfaceCoupling<2, 2, float>;
// template class InterfaceCoupling<3, 1, float>;
// template class InterfaceCoupling<3, 3, float>;

template class InterfaceCoupling<2, 1, double>;
template class InterfaceCoupling<2, 2, double>;
template class InterfaceCoupling<3, 1, double>;
template class InterfaceCoupling<3, 3, double>;

} // namespace ExaDG
