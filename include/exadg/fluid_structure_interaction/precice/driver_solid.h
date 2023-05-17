/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_

// ExaDG
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>

namespace ExaDG
{
namespace FSI
{
namespace preCICE
{
template<int dim, typename Number>
class DriverSolid : public Driver<dim, Number>
{
private:
  using VectorType = typename dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  DriverSolid(std::string const &                           input_file,
              MPI_Comm const &                              comm,
              std::shared_ptr<ApplicationBase<dim, Number>> app,
              bool const                                    is_test)
    : Driver<dim, Number>(input_file, comm, app, is_test)
  {
    structure = std::make_shared<SolverStructure<dim, Number>>();
  }


  void
  setup_application()
  {
    dealii::Timer timer_local;
    timer_local.restart();

    this->application->structure->setup();

    this->timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
  }


  void
  setup_structure()
  {
    dealii::Timer timer_local;
    timer_local.restart();

    structure->setup(this->application->structure, this->mpi_comm, this->is_test);

    this->timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }



  void
  setup_interface_coupling()
  {
    this->precice =
      std::make_shared<ExaDG::preCICE::Adapter<dim, dim, VectorType>>(this->precice_parameters,
                                                                      this->mpi_comm);

    // structure to fluid
    {
      // TODO generalize interface handling for multiple interface IDs
      this->precice->add_write_surface(
        this->application->structure->get_boundary_descriptor()->neumann_cached_bc.begin()->first,
        this->precice_parameters.write_mesh_name,
        {this->precice_parameters.displacement_data_name,
         this->precice_parameters.velocity_data_name},
        this->precice_parameters.write_data_type,
        structure->matrix_free,
        structure->pde_operator->get_dof_index(),
        dealii::numbers::invalid_unsigned_int);
    }

    // fluid to structure
    {
      this->precice->add_read_surface(
        structure->matrix_free,
        structure->pde_operator->get_container_interface_data_neumann(),
        this->precice_parameters.read_mesh_name,
        {this->precice_parameters.stress_data_name});
    }

    // initialize preCICE with initial displacement data
    VectorType displacement_structure;
    structure->pde_operator->initialize_dof_vector(displacement_structure);
    this->precice->initialize_precice(displacement_structure);
  }



  void
  setup() override
  {
    dealii::Timer timer;
    timer.restart();

    this->pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

    setup_application();

    setup_structure();

    setup_interface_coupling();

    this->timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
  }



  void
  solve() const final
  {
    bool is_new_time_window = true;
    // preCICE dictates when the time loop is finished
    while(this->precice->is_coupling_ongoing())
    {
      structure->time_integrator->advance_one_timestep_pre_solve(is_new_time_window);

      this->precice->save_current_state_if_required([&]() {});

      // update stress boundary condition for solid
      coupling_fluid_to_structure();

      // solve structural problem
      // store_solution needs to be true for compatibility
      structure->time_integrator->advance_one_timestep_partitioned_solve(is_new_time_window);
      // send displacement data to ale
      coupling_structure_to_ale(structure->time_integrator->get_displacement_np(),
                                structure->time_integrator->get_time_step_size());

      // send velocity boundary condition to fluid
      coupling_structure_to_fluid(structure->time_integrator->get_velocity_np(),
                                  structure->time_integrator->get_time_step_size());

      // TODO: Add synchronization for the time-step size here. For now, we only allow a constant
      // time-step size
      dealii::Timer precice_timer;
      this->precice->advance(structure->time_integrator->get_time_step_size());
      is_new_time_window = this->precice->is_time_window_complete();
      this->timer_tree.insert({"FSI", "preCICE"}, precice_timer.wall_time());

      // Needs to be called before the swaps in post_solve
      this->precice->reload_old_state_if_required([&]() {});

      if(is_new_time_window)
        structure->time_integrator->advance_one_timestep_post_solve();
    }
  }

  void
  print_performance_results(double const total_time) const override
  {
    this->pcout
      << std::endl
      << "_________________________________________________________________________________"
      << std::endl
      << std::endl;

    this->pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

    this->pcout << std::endl << "Structure:" << std::endl;
    structure->time_integrator->print_iterations();

    // wall times
    this->pcout << std::endl << "Wall times:" << std::endl;

    this->timer_tree.insert({"FSI"}, total_time);

    this->timer_tree.insert({"FSI"}, structure->time_integrator->get_timings(), "Structure");

    this->pcout << std::endl << "Timings for level 1:" << std::endl;
    this->timer_tree.print_level(this->pcout, 1);

    this->pcout << std::endl << "Timings for level 2:" << std::endl;
    this->timer_tree.print_level(this->pcout, 2);

    // Throughput in DoFs/s per time step per core
    dealii::types::global_dof_index DoFs = structure->pde_operator->get_number_of_dofs();

    unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(this->mpi_comm);

    dealii::Utilities::MPI::MinMaxAvg total_time_data =
      dealii::Utilities::MPI::min_max_avg(total_time, this->mpi_comm);
    double const total_time_avg = total_time_data.avg;

    unsigned int N_time_steps = structure->time_integrator->get_number_of_time_steps();

    print_throughput_unsteady(this->pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

    // computational costs in CPUh
    print_costs(this->pcout, total_time_avg, N_mpi_processes);

    this->pcout
      << "_________________________________________________________________________________"
      << std::endl
      << std::endl;
  }

private:
  void
  coupling_structure_to_ale(VectorType const & displacement_structure,
                            double const       time_step_size) const
  {
    this->precice->write_data(this->precice_parameters.write_mesh_name,
                              this->precice_parameters.displacement_data_name,
                              displacement_structure,
                              time_step_size);
  }

  void
  coupling_structure_to_fluid(VectorType const & velocity_structure,
                              double const       time_step_size) const
  {
    this->precice->write_data(this->precice_parameters.write_mesh_name,
                              this->precice_parameters.velocity_data_name,
                              velocity_structure,
                              time_step_size);
  }

  void
  coupling_fluid_to_structure() const
  {
    this->precice->read_block_data(this->precice_parameters.read_mesh_name,
                                   this->precice_parameters.stress_data_name);
  }

  // the solver
  std::shared_ptr<SolverStructure<dim, Number>> structure;
};

} // namespace preCICE
} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_ */
