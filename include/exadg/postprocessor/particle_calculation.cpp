/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// C/C++
#include <fstream>

// DEALII
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

// ExaDG
#include <exadg/postprocessor/particle_calculation.h>
#include <exadg/utilities/create_directories.h>



namespace ExaDG
{
template<int dim, typename Number>
ParticleCalculator<dim, Number>::ParticleCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), old_time(0.0)
{
}

template<int dim, typename Number>
void
ParticleCalculator<dim, Number>::setup(dealii::DoFHandler<dim> const & dof_handler_in,
                                       dealii::Mapping<dim> const &    mapping_in,
                                       ParticleData const &            data_in)
{
  if(data_in.time_control_data.is_active)
  {
    data = data_in;
    time_control.setup(data.time_control_data);
    create_directories(data.directory, mpi_comm);

    dof_handler = &dof_handler_in;
    mapping     = &mapping_in;

    // init particle handler
    particle_handler.initialize(dof_handler->get_triangulation(), *mapping);

    // get bounding boxes to distribute globally
    auto const my_bounding_box = dealii::GridTools::compute_mesh_predicate_bounding_box(
      dof_handler->get_triangulation(), dealii::IteratorFilters::LocallyOwnedCell());
    global_bounding_boxes = dealii::Utilities::MPI::all_gather(mpi_comm, my_bounding_box);

    // insert particles
    if constexpr(dim == 2)
      particle_handler.insert_global_particles(data.starting_points_2d, global_bounding_boxes);
    else if constexpr(dim == 3)
      particle_handler.insert_global_particles(data.starting_points_3d, global_bounding_boxes);

    // reserve space for the velocities in the cell
    solution_values.reinit(dof_handler->get_fe().n_dofs_per_cell());

    // deal with lost particles
    particle_handler.signals.particle_lost.connect(
      [this](const typename dealii::Particles::ParticleIterator<dim> &         particle,
             const typename dealii::Triangulation<dim>::active_cell_iterator & cell) {
        this->track_lost_particle(particle, cell);
      });
  }
}


template<int dim, typename Number>
void
ParticleCalculator<dim, Number>::track_lost_particle(
  const typename dealii::Particles::ParticleIterator<dim> &         particle,
  const typename dealii::Triangulation<dim>::active_cell_iterator & cell)
{
  (void)particle;
  (void)cell;
  ++lost_paticles;
}

template<int dim, typename Number>
void
ParticleCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                          double const       time,
                                          bool const         print_output)
{
  double const dt = time - old_time;
  old_time        = time;

  dealii::FEPointEvaluation<dim, dim, dim, Number> evaluator(*mapping,
                                                             dof_handler->get_fe(),
                                                             dealii::update_values);

  // go over all particles
  auto particle = particle_handler.begin();
  while(particle != particle_handler.end())
  {
    // get cell in which the current particle is
    auto const cell    = particle->get_surrounding_cell();
    auto const dh_cell = typename dealii::DoFHandler<dim>::cell_iterator(*cell, dof_handler);

    // get velocity in the cell
    solution_values = 0.;
    dh_cell->get_dof_values(velocity, solution_values);

    // collect all particles in the cell
    auto const pic = particle_handler.particles_in_cell(cell);

    // collect current reference position of particles
    std::vector<dealii::Point<dim>> particle_positions;
    for(auto const & p : pic)
      particle_positions.push_back(p.get_reference_location());

    // evaluate the velocities at the positions of the particles
    evaluator.reinit(cell, particle_positions);
    evaluator.evaluate(make_array_view(solution_values), dealii::EvaluationFlags::values);

    // update position of the particles
    for(unsigned int particle_index = 0; particle != pic.end(); ++particle, ++particle_index)
    {
      particle->set_location(particle->get_location() + dt * evaluator.get_value(particle_index));
    }
  }

  // sort particles into new cells
  particle_handler.sort_particles_into_subdomains_and_cells();


  // write output file
  if(print_output)
  {
    dealii::Particles::DataOut<dim> data_out;
    data_out.build_patches(particle_handler);
    data_out.write_vtu_with_pvtu_record(data.directory,
                                        data.filename + "_particles",
                                        time_control.get_counter(),
                                        mpi_comm);
  }
}

template class ParticleCalculator<2, float>;
template class ParticleCalculator<2, double>;

template class ParticleCalculator<3, float>;
template class ParticleCalculator<3, double>;

} // namespace ExaDG
