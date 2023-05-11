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

// deal.II
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>

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

    map_evaluator[quad_index].reinit(interface_data_dst->get_array_q_points(quad_index),
                                     dof_handler_src_.get_triangulation(),
                                     mapping_src_);

    if(not map_evaluator[quad_index].all_points_found())
    {
      std::string const file_name =
        "interface_coupling_quad_index_" + dealii::Utilities::to_string(quad_index);
      plot_points_and_triangulation(interface_data_dst->get_array_q_points(quad_index),
                                    dof_handler_src_,
                                    mapping_src_,
                                    file_name);
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Setup of InterfaceCoupling was not successful. Not all points have been found."));
    }
  }
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::plot_points_and_triangulation(
  std::vector<dealii::Point<dim>> const & points,
  dealii::DoFHandler<dim> const &         dof_handler,
  dealii::Mapping<dim> const &            mapping,
  std::string const &                     file_name) const
{
  // higher order mapped triangulation output
  dealii::DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  dealii::DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  data_out.build_patches(mapping, 4, dealii::DataOut<dim>::curved_inner_cells);
  data_out.write_vtu_with_pvtu_record(
    "./", file_name + "_grid", 0, dof_handler.get_communicator(), 4);

  // point output via particle handler
  dealii::Triangulation<dim> particle_dummy_tria;
  double                     min_coord = points[0][0];
  double                     max_coord = points[0][0];
  for(auto const & p : points)
  {
    for(unsigned int d = 0; d < dim; ++d)
    {
      min_coord = p[d] < min_coord ? p[d] : min_coord;
      max_coord = p[d] > max_coord ? p[d] : max_coord;
    }
  }
  min_coord = dealii::Utilities::MPI::min(min_coord, dof_handler.get_communicator());
  max_coord = dealii::Utilities::MPI::max(max_coord, dof_handler.get_communicator());
  dealii::GridGenerator::hyper_cube(particle_dummy_tria, min_coord, max_coord);
  dealii::MappingQGeneric<dim>                 particle_dummy_mapping(1 /* mapping_degree */);
  dealii::Particles::ParticleHandler<dim, dim> particle_handler(particle_dummy_tria,
                                                                particle_dummy_mapping);

  particle_handler.insert_particles(points);

  dealii::Particles::DataOut<dim, dim> particle_output;
  particle_output.build_patches(particle_handler);
  particle_output.write_vtu_with_pvtu_record("./",
                                             file_name + "_points",
                                             0,
                                             dof_handler.get_communicator());
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
