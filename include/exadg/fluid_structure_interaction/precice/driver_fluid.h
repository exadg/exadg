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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_FLUID_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_FLUID_H_

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

// utilities
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

// grid
#include <exadg/grid/grid_motion_elasticity.h>
#include <exadg/grid/grid_motion_poisson.h>
#include <exadg/poisson/spatial_discretization/operator.h>

// IncNS
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

namespace ExaDG
{
namespace FSI
{
namespace preCICE
{
template<int dim, typename Number>
class DriverFluid : public Driver<dim, Number>
{
private:
  using VectorType = typename dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  DriverFluid(std::string const &                           input_file,
              MPI_Comm const &                              comm,
              std::shared_ptr<ApplicationBase<dim, Number>> app,
              bool const                                    is_test)
    : Driver<dim, Number>(input_file, comm, app, is_test)
  {
    fluid = std::make_shared<SolverFluid<dim, Number>>();
  }

  void
  setup_application()
  {
    dealii::Timer timer_local;
    timer_local.restart();

    this->application->fluid->setup();

    this->timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
  }


  void
  setup_fluid_and_ale()
  {
    dealii::Timer timer_local;

    fluid->setup(this->application->fluid, this->mpi_comm, this->is_test);

    this->timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }


  void
  setup_interface_coupling()
  {
    this->precice =
      std::make_shared<ExaDG::preCICE::Adapter<dim, dim, VectorType>>(this->precice_parameters,
                                                                      this->mpi_comm);

    // fluid to structure
    {
      // TODO generalize interface handling for multiple interface IDs
      this->precice->add_write_surface(this->application->fluid->get_boundary_descriptor()
                                         ->velocity->dirichlet_cached_bc.begin()
                                         ->first,
                                       this->precice_parameters.write_mesh_name,
                                       {this->precice_parameters.stress_data_name},
                                       this->precice_parameters.write_data_type,
                                       fluid->matrix_free,
                                       fluid->pde_operator->get_dof_index_velocity(),
                                       fluid->pde_operator->get_quad_index_velocity_linear());
    }

    // structure to ALE
    {
      // Poisson mesh movement
      if(this->application->fluid->get_parameters().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        std::shared_ptr<Poisson::DeformedMapping<dim, Number>> poisson_ale_mapping =
          std::dynamic_pointer_cast<Poisson::DeformedMapping<dim, Number>>(fluid->ale_mapping);

        this->precice->add_read_surface(
          poisson_ale_mapping->get_matrix_free(),
          poisson_ale_mapping->get_pde_operator()->get_container_interface_data(),
          this->precice_parameters.ale_mesh_name,
          {this->precice_parameters.displacement_data_name});
      }
      // Elasticity mesh movement
      else if(this->application->fluid->get_parameters().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        std::shared_ptr<Structure::DeformedMapping<dim, Number>> structure_ale_mapping =
          std::dynamic_pointer_cast<Structure::DeformedMapping<dim, Number>>(fluid->ale_mapping);

        this->precice->add_read_surface(
          structure_ale_mapping->get_matrix_free(),
          structure_ale_mapping->get_pde_operator()->get_container_interface_data_dirichlet(),
          this->precice_parameters.ale_mesh_name,
          {this->precice_parameters.displacement_data_name});
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
    }

    // structure to fluid
    {
      this->precice->add_read_surface(fluid->matrix_free,
                                      fluid->pde_operator->get_container_interface_data(),
                                      this->precice_parameters.read_mesh_name,
                                      {this->precice_parameters.velocity_data_name});
    }

    // initialize preCICE with initial stress data
    VectorType initial_stress;
    fluid->pde_operator->initialize_vector_velocity(initial_stress);
    this->precice->initialize_precice(initial_stress);
  }



  void
  setup() override
  {
    dealii::Timer timer;
    timer.restart();

    this->pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

    setup_application();

    setup_fluid_and_ale();

    setup_interface_coupling();

    this->timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
  }



  void
  solve() const final
  {
    Assert(this->application->fluid->get_parameters().adaptive_time_stepping == false,
           dealii::ExcNotImplemented());

    bool is_new_time_window = true;
    // preCICE dictates when the time loop is finished
    while(this->precice->is_coupling_ongoing())
    {
      // pre-solve
      fluid->time_integrator->advance_one_timestep_pre_solve(is_new_time_window);

      this->precice->save_current_state_if_required([&]() {});

      {
        coupling_structure_to_ale();

        // move the fluid mesh and update dependent data structures
        fluid->solve_ale();

        // update velocity boundary condition for fluid
        coupling_structure_to_fluid();

        // solve fluid problem
        fluid->time_integrator->advance_one_timestep_partitioned_solve(is_new_time_window);

        // compute and send stress to solid
        coupling_fluid_to_structure();

        // TODO: Add synchronization for the time-step size here. For now, we only allow a constant
        // time-step size
        dealii::Timer precice_timer;
        this->precice->advance(fluid->time_integrator->get_time_step_size());
        is_new_time_window = this->precice->is_time_window_complete();
        this->timer_tree.insert({"FSI", "preCICE"}, precice_timer.wall_time());
      }

      // Needs to be called before the swaps in post_solve
      this->precice->reload_old_state_if_required([&]() {});

      // post-solve
      if(is_new_time_window)
        fluid->time_integrator->advance_one_timestep_post_solve();
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

    this->pcout << std::endl << "Fluid:" << std::endl;
    fluid->time_integrator->print_iterations();

    this->pcout << std::endl << "ALE:" << std::endl;
    fluid->ale_mapping->print_iterations();

    // wall times
    this->pcout << std::endl << "Wall times:" << std::endl;

    this->timer_tree.insert({"FSI"}, total_time);

    this->timer_tree.insert({"FSI"}, fluid->time_integrator->get_timings(), "Fluid");

    this->pcout << std::endl << "Timings for level 1:" << std::endl;
    this->timer_tree.print_level(this->pcout, 1);

    this->pcout << std::endl << "Timings for level 2:" << std::endl;
    this->timer_tree.print_level(this->pcout, 2);

    // Throughput in DoFs/s per time step per core
    dealii::types::global_dof_index DoFs = fluid->pde_operator->get_number_of_dofs();

    if(this->application->fluid->get_parameters().mesh_movement_type ==
       IncNS::MeshMovementType::Poisson)
    {
      std::shared_ptr<Poisson::DeformedMapping<dim, Number>> poisson_ale_mapping =
        std::dynamic_pointer_cast<Poisson::DeformedMapping<dim, Number>>(fluid->ale_mapping);

      DoFs += poisson_ale_mapping->get_pde_operator()->get_number_of_dofs();
    }
    else if(this->application->fluid->get_parameters().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      std::shared_ptr<Structure::DeformedMapping<dim, Number>> elasticity_ale_mapping =
        std::dynamic_pointer_cast<Structure::DeformedMapping<dim, Number>>(fluid->ale_mapping);

      DoFs += elasticity_ale_mapping->get_pde_operator()->get_number_of_dofs();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(this->mpi_comm);

    dealii::Utilities::MPI::MinMaxAvg total_time_data =
      dealii::Utilities::MPI::min_max_avg(total_time, this->mpi_comm);
    double const total_time_avg = total_time_data.avg;

    unsigned int N_time_steps = fluid->time_integrator->get_number_of_time_steps();

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
  coupling_structure_to_ale() const
  {
    // TODO: parametrize names
    this->precice->read_block_data(this->precice_parameters.ale_mesh_name,
                                   this->precice_parameters.displacement_data_name);
  }

  void
  coupling_structure_to_fluid() const
  {
    // TODO: parametrize names
    this->precice->read_block_data(this->precice_parameters.read_mesh_name,
                                   this->precice_parameters.velocity_data_name);
  }

  void
  coupling_fluid_to_structure() const
  {
    VectorType stress_fluid;
    fluid->pde_operator->initialize_vector_velocity(stress_fluid);
    // calculate fluid stress at fluid-structure interface
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity_np(),
                                               fluid->time_integrator->get_pressure_np());
    stress_fluid *= -1.0;
    this->precice->write_data(this->precice_parameters.write_mesh_name,
                              this->precice_parameters.stress_data_name,
                              stress_fluid,
                              fluid->time_integrator->get_time_step_size());
  }

  // solver
  std::shared_ptr<SolverFluid<dim, Number>> fluid;
};

} // namespace preCICE
} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_ */
