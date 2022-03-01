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

// application
#include <exadg/fluid_structure_interaction/precice/interface_coupling.h>
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

// utilities
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>

// grid
#include <exadg/grid/grid_motion_elasticity.h>
#include <exadg/grid/grid_motion_poisson.h>
#include <exadg/poisson/spatial_discretization/operator.h>

// Structure
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>

namespace ExaDG
{
namespace FSI
{
namespace preCICE
{
using namespace dealii;

template<int dim, typename Number>
class DriverSolid : public Driver<dim, Number>
{
private:
  using VectorType = typename LinearAlgebra::distributed::Vector<Number>;

public:
  DriverSolid(std::string const &                           input_file,
              MPI_Comm const &                              comm,
              std::shared_ptr<ApplicationBase<dim, Number>> app,
              bool const                                    is_test)
    : Driver<dim, Number>(input_file, comm, app, is_test)
  {
  }

  void
  setup()
  {
    dealii::Timer timer;
    timer.restart();

    this->pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

    /************************************** APPLICATION *****************************************/
    {
      dealii::Timer timer_local;
      timer_local.restart();

      this->application->setup_structure();

      this->timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
    }
    /************************************** APPLICATION *****************************************/

    /**************************************** STRUCTURE *****************************************/
    {
      dealii::Timer timer_local;
      timer_local.restart();

      // setup spatial operator
      structure_operator = std::make_shared<Structure::Operator<dim, Number>>(
        this->application->structure->get_grid(),
        this->application->structure->get_boundary_descriptor(),
        this->application->structure->get_field_functions(),
        this->application->structure->get_material_descriptor(),
        this->application->structure->get_parameters(),
        "elasticity",
        this->mpi_comm);

      // initialize matrix_free
      structure_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
      structure_matrix_free_data->append(structure_operator);

      structure_matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
      structure_matrix_free->reinit(*this->application->structure->get_grid()->mapping,
                                    structure_matrix_free_data->get_dof_handler_vector(),
                                    structure_matrix_free_data->get_constraint_vector(),
                                    structure_matrix_free_data->get_quadrature_vector(),
                                    structure_matrix_free_data->data);

      structure_operator->setup(structure_matrix_free, structure_matrix_free_data);

      // initialize postprocessor
      structure_postprocessor = this->application->structure->create_postprocessor();
      structure_postprocessor->setup(structure_operator->get_dof_handler(),
                                     *this->application->structure->get_grid()->mapping);

      this->timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
    }

    /**************************************** STRUCTURE *****************************************/


    /*********************************** INTERFACE COUPLING *************************************/

    this->precice =
      std::make_shared<ExaDG::preCICE::Adapter<dim, dim, VectorType>>(this->precice_parameters,
                                                                      this->mpi_comm);

    this->precice->add_write_surface(
      this->application->structure->get_boundary_descriptor()->neumann_cached_bc.begin()->first,
      this->precice_parameters.write_mesh_name,
      {this->precice_parameters.displacement_data_name,
       this->precice_parameters.velocity_data_name},
      this->precice_parameters.write_data_type,
      structure_matrix_free,
      structure_operator->get_dof_index(),
      numbers::invalid_unsigned_int);

    {
      std::vector<unsigned int> quad_indices;
      quad_indices.emplace_back(structure_operator->get_quad_index());

      // VectorType stress_fluid;
      auto exadg_terminal = std::make_shared<ExaDG::preCICE::InterfaceCoupling<dim, dim, Number>>();
      auto quadrature_point_locations = exadg_terminal->setup(
        structure_matrix_free,
        structure_operator->get_dof_index(),
        quad_indices,
        this->application->structure->get_boundary_descriptor()->neumann_cached_bc);

      this->precice->add_read_surface(quadrature_point_locations,
                                      structure_matrix_free,
                                      exadg_terminal,
                                      this->precice_parameters.read_mesh_name,
                                      {this->precice_parameters.stress_data_name});

      VectorType displacement_structure;
      structure_operator->initialize_dof_vector(displacement_structure);
      displacement_structure = 0;
      this->precice->initialize_precice(displacement_structure);
    }

    /*********************************** INTERFACE COUPLING *************************************/


    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

    // Structure
    {
      // initialize time integrator
      structure_time_integrator = std::make_shared<Structure::TimeIntGenAlpha<dim, Number>>(
        structure_operator,
        structure_postprocessor,
        this->application->structure->get_parameters(),
        this->mpi_comm,
        this->is_test);

      structure_time_integrator->setup(
        this->application->structure->get_parameters().restarted_simulation);

      structure_operator->setup_solver();
    }

    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/
  }

  void
  solve() const final
  {
    bool is_new_time_window = true;
    // preCICE dictates when the time loop is finished
    while(this->precice->is_coupling_ongoing())
    {
      structure_time_integrator->advance_one_timestep_pre_solve(is_new_time_window);

      this->precice->save_current_state_if_required([&]() {});

      // update stress boundary condition for solid
      coupling_fluid_to_structure();

      // solve structural problem
      // store_solution needs to be true for compatibility
      structure_time_integrator->advance_one_timestep_partitioned_solve(is_new_time_window, true);
      // send displacement data to ale
      coupling_structure_to_ale(structure_time_integrator->get_displacement_np(),
                                structure_time_integrator->get_time_step_size());

      // send velocity boundary condition for fluid
      coupling_structure_to_fluid(structure_time_integrator->get_velocity_np(),
                                  structure_time_integrator->get_time_step_size());

      // TODO: Add synchronization for the time-step size here. For now, we only allow a constant
      // time-step size
      dealii::Timer precice_timer;
      this->precice->advance(structure_time_integrator->get_time_step_size());
      is_new_time_window = this->precice->is_time_window_complete();
      this->timer_tree.insert({"FSI", "preCICE"}, precice_timer.wall_time());

      // Needs to be called before the swaps in post_solve
      this->precice->reload_old_state_if_required([&]() {});

      if(is_new_time_window)
        structure_time_integrator->advance_one_timestep_post_solve();
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
    structure_time_integrator->print_iterations();

    // wall times
    this->pcout << std::endl << "Wall times:" << std::endl;

    this->timer_tree.insert({"FSI"}, total_time);

    this->timer_tree.insert({"FSI"}, structure_time_integrator->get_timings(), "Structure");

    this->pcout << std::endl << "Timings for level 1:" << std::endl;
    this->timer_tree.print_level(this->pcout, 1);

    this->pcout << std::endl << "Timings for level 2:" << std::endl;
    // this->timer_tree.print_level(this->pcout, 2);

    // Throughput in DoFs/s per time step per core
    types::global_dof_index DoFs = structure_operator->get_number_of_dofs();

    unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(this->mpi_comm);

    Utilities::MPI::MinMaxAvg total_time_data =
      Utilities::MPI::min_max_avg(total_time, this->mpi_comm);
    double const total_time_avg = total_time_data.avg;

    unsigned int N_time_steps = structure_time_integrator->get_number_of_time_steps();

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
                            const double       time_step_size) const
  {
    this->precice->write_data(this->precice_parameters.write_mesh_name,
                              this->precice_parameters.displacement_data_name,
                              displacement_structure,
                              time_step_size);
  }

  void
  coupling_structure_to_fluid(VectorType const & velocity_structure,
                              const double       time_step_size) const
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

  // grid
  std::shared_ptr<Grid<dim>> structure_grid;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> structure_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     structure_matrix_free;

  // spatial discretization
  std::shared_ptr<Structure::Operator<dim, Number>> structure_operator;

  // temporal discretization
  std::shared_ptr<Structure::TimeIntGenAlpha<dim, Number>> structure_time_integrator;

  // postprocessor
  std::shared_ptr<Structure::PostProcessor<dim, Number>> structure_postprocessor;

  /**************************************** STRUCTURE *****************************************/
};

} // namespace preCICE
} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_ */
