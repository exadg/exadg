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
#include <exadg/postprocessor/write_output.h>

namespace ExaDG
{
template<int rank, int dim, typename Number>
InterfaceCoupling<rank, dim, Number>::InterfaceCoupling() : dof_handler_src(nullptr)
{
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::setup(
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>> interface_data_dst_,
  dealii::DoFHandler<dim> const &                            dof_handler_src_,
  dealii::Mapping<dim> const &                               mapping_src_,
  std::vector<bool> const &                                  marked_vertices_src_,
  double const                                               tolerance_)
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

    auto const * points = &interface_data_dst->get_array_q_points(quad_index);

    map_evaluator[quad_index].reinit(*points, dof_handler_src_.get_triangulation(), mapping_src_);

    if(not map_evaluator[quad_index].all_points_found())
    {
      // get vector of points not found
      std::vector<dealii::Point<dim>> points_not_found;
      points_not_found.reserve(points->size());
      unsigned int n_points_not_found = 0;
      for(unsigned int i = 0; i < points->size(); ++i)
      {
        if(not map_evaluator[quad_index].point_found(i))
        {
          n_points_not_found += 1;
          points_not_found.push_back((*points)[i]);
        }
      }
      n_points_not_found =
        dealii::Utilities::MPI::sum(n_points_not_found, dof_handler_src->get_communicator());

      std::string const file_name =
        "interface_coupling_quad_index_" + dealii::Utilities::to_string(quad_index);

      write_grid(dof_handler_src->get_triangulation(),
                 mapping_src_,
                 4,
                 "./",
                 file_name,
                 0,
                 dof_handler_src->get_communicator());

      write_points(points_not_found, "./", file_name, 0, dof_handler_src->get_communicator());

      AssertThrow(map_evaluator[quad_index].all_points_found(),
                  dealii::ExcMessage(std::string("Setup of InterfaceCoupling was not successful: " +
                                                 std::to_string(n_points_not_found) +
                                                 " points have not been found.")));
    }
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

template class InterfaceCoupling<0, 2, float>;
template class InterfaceCoupling<1, 2, float>;
template class InterfaceCoupling<0, 3, float>;
template class InterfaceCoupling<1, 3, float>;

template class InterfaceCoupling<0, 2, double>;
template class InterfaceCoupling<1, 2, double>;
template class InterfaceCoupling<0, 3, double>;
template class InterfaceCoupling<1, 3, double>;

} // namespace ExaDG
