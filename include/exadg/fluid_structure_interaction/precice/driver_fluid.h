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
        this->precice->add_read_surface(fluid->ale_matrix_free,
                                        fluid->ale_poisson_operator->get_container_interface_data(),
                                        this->precice_parameters.ale_mesh_name,
                                        {this->precice_parameters.displacement_data_name});
      }
      // Elasticity mesh movement
      else if(this->application->fluid->get_parameters().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        this->precice->add_read_surface(
          fluid->ale_matrix_free,
          fluid->ale_elasticity_operator->get_container_interface_data_dirichlet(),
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
    this->allowed_time_step_size = this->precice->initialize_precice(initial_stress);
  }



  void
  setup()
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
    bool is_new_time_window = true;
    fluid->time_integrator->set_current_time_step_size(
      std::min(this->allowed_time_step_size, fluid->time_integrator->get_time_step_size()));
    // preCICE dictates when the time loop is finished
    while(this->precice->is_coupling_ongoing())
    {
      // pre-solve
      fluid->time_integrator->advance_one_timestep_pre_solve(is_new_time_window);

      this->precice->save_current_state_if_required([&]() {});

      {
        coupling_structure_to_ale();

        // move the fluid mesh and update dependent data structures
        fluid->solve_ale(this->application->fluid, this->is_test);

        // update velocity boundary condition for fluid
        coupling_structure_to_fluid();

        // solve fluid problem
        fluid->time_integrator->advance_one_timestep_partitioned_solve(is_new_time_window);

        // compute and send stress to solid
        coupling_fluid_to_structure();

        dealii::Timer precice_timer;
        this->allowed_time_step_size =
          this->precice->advance(fluid->time_integrator->get_time_step_size());
        is_new_time_window = this->precice->is_time_window_complete();
        this->timer_tree.insert({"FSI", "preCICE"}, precice_timer.wall_time());
      }

      // Needs to be called before the swaps in post_solve
      this->precice->reload_old_state_if_required([&]() {});

      // post-solve
      if(is_new_time_window)
      {
        // computes new time-step size
        fluid->time_integrator->advance_one_timestep_post_solve();
        // next, we synchronize the time-step sizes. Subcycling would in theory be compatible in
        // explicit coupling schemes here. In implicit coupling schemes, we need matching
        // time-window sizes (either constant (serial or parallel schemes) or adaptively (serial
        // scheme)) as the time-step size push back happens in the
        // 'advance_one_timestep_post_solve()' and we cannot change two subsequent time-step sizes
        // without a push back operation, otherwise we falsify the time integrator. However, the
        // treatment of the boundary conditions is not handled correctly when using subcycling, as
        // we apply the 'whole displacement' within the first time-step. In order to cope with
        // subcycling, either the boundary condition needs to be adopted or we need to wait for a
        // newer preCICE version (currently v2.3.0) which can handle this.
        // Hence, we do not adjust the time-step-size of the fluid solver, but rather Assert that it
        // is still valid. Note that we would run into a preCICE error otherwise reporting the same
        // issue otherwise.
        // fluid->time_integrator->set_current_time_step_size(
        //   std::min(this->allowed_time_step_size, fluid->time_integrator->get_time_step_size()));
        Assert(
          fluid->time_integrator->get_time_step_size() <
            this->allowed_time_step_size + std::numeric_limits<double>::min(),
          dealii::ExcMessage(
            "The solver time-step size exceeded the maximum admissible time-step size allowed by preCICE. "
            "If you select adaptive time-stepping, make sure to let the Fluid participant steer the time-window size. "
            "In any other case, please disable adaptive time-stepping."));
      }
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
    fluid->ale_grid_motion->print_iterations();

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
      DoFs += fluid->ale_poisson_operator->get_number_of_dofs();
    }
    else if(this->application->fluid->get_parameters().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      DoFs += fluid->ale_elasticity_operator->get_number_of_dofs();
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
