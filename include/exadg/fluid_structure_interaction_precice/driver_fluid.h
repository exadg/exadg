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
#include <exadg/fluid_structure_interaction_precice/interface_coupling.h>

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
using namespace dealii;

template<int dim, typename Number>
class DriverFluid : public Driver<dim, Number>
{
private:
  using VectorType = typename LinearAlgebra::distributed::Vector<Number>;

public:
  DriverFluid(std::string const &                           input_file,
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

      this->application->setup();

      this->timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
    }
    /************************************** APPLICATION *****************************************/

    /******************************************* ALE ********************************************/
    {
      dealii::Timer timer_local;
      timer_local.restart();

      // ALE: initialize PDE operator
      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        ale_poisson_operator = std::make_shared<Poisson::Operator<dim, Number, dim>>(
          this->application->get_grid_fluid(),
          this->application->get_boundary_descriptor_ale_poisson(),
          this->application->get_field_functions_ale_poisson(),
          this->application->get_parameters_ale_poisson(),
          "Poisson",
          this->mpi_comm);
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        ale_elasticity_operator = std::make_shared<Structure::Operator<dim, Number>>(
          this->application->get_grid_fluid(),
          this->application->get_boundary_descriptor_ale_elasticity(),
          this->application->get_field_functions_ale_elasticity(),
          this->application->get_material_descriptor_ale_elasticity(),
          this->application->get_parameters_ale_elasticity(),
          "ale_elasticity",
          this->mpi_comm);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      // ALE: initialize matrix_free_data
      ale_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();

      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        if(this->application->get_parameters_ale_poisson().enable_cell_based_face_loops)
          Categorization::do_cell_based_loops(*this->application->get_grid_fluid()->triangulation,
                                              ale_matrix_free_data->data);

        ale_matrix_free_data->append(ale_poisson_operator);
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        ale_matrix_free_data->append(ale_elasticity_operator);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      // ALE: initialize matrix_free
      ale_matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
      ale_matrix_free->reinit(*this->application->get_grid_fluid()->mapping,
                              ale_matrix_free_data->get_dof_handler_vector(),
                              ale_matrix_free_data->get_constraint_vector(),
                              ale_matrix_free_data->get_quadrature_vector(),
                              ale_matrix_free_data->data);

      // ALE: setup PDE operator and solver
      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        ale_poisson_operator->setup(ale_matrix_free, ale_matrix_free_data);
        ale_poisson_operator->setup_solver();
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        ale_elasticity_operator->setup(ale_matrix_free, ale_matrix_free_data);
        ale_elasticity_operator->setup_solver();
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      // ALE: create grid motion object
      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        fluid_grid_motion = std::make_shared<GridMotionPoisson<dim, Number>>(
          this->application->get_grid_fluid()->mapping, ale_poisson_operator);
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        fluid_grid_motion = std::make_shared<GridMotionElasticity<dim, Number>>(
          this->application->get_grid_fluid()->mapping,
          ale_elasticity_operator,
          this->application->get_parameters_ale_elasticity());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      this->timer_tree.insert({"FSI", "Setup", "ALE"}, timer_local.wall_time());
    }
    /******************************************* ALE ********************************************/

    /****************************************** FLUID *******************************************/
    {
      dealii::Timer timer_local;
      timer_local.restart();

      // initialize fluid_operator
      fluid_operator =
        IncNS::create_operator<dim, Number>(this->application->get_grid_fluid(),
                                            fluid_grid_motion,
                                            this->application->get_boundary_descriptor_fluid(),
                                            this->application->get_field_functions_fluid(),
                                            this->application->get_parameters_fluid(),
                                            "fluid",
                                            this->mpi_comm);

      // initialize matrix_free
      fluid_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
      fluid_matrix_free_data->append(fluid_operator);

      fluid_matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
      if(this->application->get_parameters_fluid().use_cell_based_face_loops)
        Categorization::do_cell_based_loops(*this->application->get_grid_fluid()->triangulation,
                                            fluid_matrix_free_data->data);
      std::shared_ptr<dealii::Mapping<dim> const> mapping =
        get_dynamic_mapping<dim, Number>(this->application->get_grid_fluid(), fluid_grid_motion);
      fluid_matrix_free->reinit(*mapping,
                                fluid_matrix_free_data->get_dof_handler_vector(),
                                fluid_matrix_free_data->get_constraint_vector(),
                                fluid_matrix_free_data->get_quadrature_vector(),
                                fluid_matrix_free_data->data);

      // setup Navier-Stokes operator
      fluid_operator->setup(fluid_matrix_free, fluid_matrix_free_data);

      // setup postprocessor
      fluid_postprocessor = this->application->create_postprocessor_fluid();
      fluid_postprocessor->setup(*fluid_operator);

      this->timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
    }
    /****************************************** FLUID *******************************************/


    /*********************************** INTERFACE COUPLING *************************************/

    // writing
    this->precice =
      std::make_shared<Adapter::Adapter<dim, dim, VectorType>>(this->precice_parameters);

    write_mesh_name = this->precice_parameters.write_mesh_name;
    write_data_name = this->precice_parameters.write_data_name;
    this->precice->add_write_interface(this->application->get_boundary_descriptor_fluid()
                                         ->velocity->dirichlet_mortar_bc.begin()
                                         ->first,
                                       this->precice_parameters.write_mesh_name,
                                       this->precice_parameters.write_data_name,
                                       "values_on_quads",
                                       fluid_matrix_free,
                                       fluid_operator->get_dof_index_velocity(),
                                       fluid_operator->get_quad_index_velocity_linear());

    // structure to ALE
    {
      if(this->application->get_parameters_fluid().mesh_movement_type ==
         IncNS::MeshMovementType::Poisson)
      {
        std::vector<unsigned int> quad_indices;
        if(this->application->get_parameters_ale_poisson().spatial_discretization ==
           Poisson::SpatialDiscretization::DG)
          quad_indices.emplace_back(ale_poisson_operator->get_quad_index());
        else if(this->application->get_parameters_ale_poisson().spatial_discretization ==
                Poisson::SpatialDiscretization::CG)
          quad_indices.emplace_back(ale_poisson_operator->get_quad_index_gauss_lobatto());
        else
          AssertThrow(false, ExcNotImplemented());

        // VectorType stress_fluid;
        communicator_ale = std::make_shared<InterfaceCoupling<dim, dim, Number>>(this->precice);
        VectorType displacement_structure;
        ale_poisson_operator->initialize_dof_vector(displacement_structure);
        communicator_ale->setup(
          ale_matrix_free,
          ale_poisson_operator->get_dof_index(),
          quad_indices,
          this->application->get_boundary_descriptor_ale_poisson()->dirichlet_mortar_bc,
          "ALE-Mesh",
          "Displacement");
      }
      else if(this->application->get_parameters_fluid().mesh_movement_type ==
              IncNS::MeshMovementType::Elasticity)
      {
        std::vector<unsigned int> quad_indices;
        quad_indices.emplace_back(ale_elasticity_operator->get_quad_index_gauss_lobatto());

        VectorType displacement_structure;
        ale_elasticity_operator->initialize_dof_vector(displacement_structure);
        communicator_ale = std::make_shared<InterfaceCoupling<dim, dim, Number>>(this->precice);
        communicator_ale->setup(
          ale_matrix_free,
          ale_elasticity_operator->get_dof_index(),
          quad_indices,
          this->application->get_boundary_descriptor_ale_elasticity()->dirichlet_mortar_bc,
          "ALE-Mesh",
          "Displacement");
      }
      else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    }

    // structure to fluid
    {
      std::vector<unsigned int> quad_indices;
      quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_linear());
      quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_nonlinear());
      quad_indices.emplace_back(fluid_operator->get_quad_index_velocity_gauss_lobatto());

      VectorType velocity_structure;
      fluid_operator->initialize_vector_velocity(velocity_structure);
      communicator_fluid = std::make_shared<InterfaceCoupling<dim, dim, Number>>(this->precice);
      communicator_fluid->setup(
        fluid_matrix_free,
        fluid_operator->get_dof_index_velocity(),
        quad_indices,
        this->application->get_boundary_descriptor_fluid()->velocity->dirichlet_mortar_bc,
        "Fluid-Mesh-read",
        "Velocity");
      // TODO: Change to Velocity

      VectorType initial_stress;
      fluid_operator->initialize_vector_velocity(initial_stress);
      initial_stress = 0;
      this->precice->initialize_precice(initial_stress);
    }


    /*********************************** INTERFACE COUPLING *************************************/


    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

    // fluid
    {
      // setup time integrator before calling setup_solvers (this is necessary since the setup
      // of the solvers depends on quantities such as the time_step_size or gamma0!!!)
      AssertThrow(this->application->get_parameters_fluid().solver_type ==
                    IncNS::SolverType::Unsteady,
                  ExcMessage("Invalid parameter in context of fluid-structure interaction."));

      // initialize fluid_time_integrator
      fluid_time_integrator =
        IncNS::create_time_integrator<dim, Number>(fluid_operator,
                                                   this->application->get_parameters_fluid(),
                                                   this->mpi_comm,
                                                   this->is_test,
                                                   fluid_postprocessor);

      fluid_time_integrator->setup(this->application->get_parameters_fluid().restarted_simulation);

      fluid_operator->setup_solvers(
        fluid_time_integrator->get_scaling_factor_time_derivative_term(),
        fluid_time_integrator->get_velocity());
    }

    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/
  }

  void
  solve() const final
  {
    Assert(this->application->get_parameters_fluid().adaptive_time_stepping == false,
           ExcNotImplemented());

    // initial true
    fluid_time_integrator->advance_one_timestep_pre_solve(true);

    // preCICE dictates when the time loop is finished
    while(this->precice->is_coupling_ongoing())
    {
      // pre-solve
      fluid_time_integrator->advance_one_timestep_pre_solve(
        this->precice->is_time_window_complete());

      this->precice->save_current_state_if_required([&]() { /*TODO*/ });

      // solve (using strongly-coupled partitioned scheme)
      {
        coupling_structure_to_ale();

        // move the fluid mesh and update dependent data structures
        solve_ale();

        // update velocity boundary condition for fluid
        coupling_structure_to_fluid();

        // solve fluid problem
        fluid_time_integrator->advance_one_timestep_partitioned_solve(false, true);

        // update stress boundary condition for solid
        coupling_fluid_to_structure();

        this->precice->advance(fluid_time_integrator->get_time_step_size());
      }

      // Needs to be called before the swaps in post_solve
      this->precice->reload_old_state_if_required([&]() { /*TODO*/ });

      // post-solve
      if(this->precice->is_time_window_complete())
        fluid_time_integrator->advance_one_timestep_post_solve();
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
    fluid_time_integrator->print_iterations();

    this->pcout << std::endl << "ALE:" << std::endl;
    fluid_grid_motion->print_iterations();

    // wall times
    this->pcout << std::endl << "Wall times:" << std::endl;

    this->timer_tree.insert({"FSI"}, total_time);

    this->timer_tree.insert({"FSI"}, fluid_time_integrator->get_timings(), "Fluid");

    this->pcout << std::endl << "Timings for level 1:" << std::endl;
    this->timer_tree.print_level(this->pcout, 1);

    // TODO: This assumes that we have at least two levels, otherwise failure
    this->pcout << std::endl << "Timings for level 2:" << std::endl;
    this->timer_tree.print_level(this->pcout, 2);

    // Throughput in DoFs/s per time step per core
    types::global_dof_index DoFs = fluid_operator->get_number_of_dofs();

    if(this->application->get_parameters_fluid().mesh_movement_type ==
       IncNS::MeshMovementType::Poisson)
    {
      DoFs += ale_poisson_operator->get_number_of_dofs();
    }
    else if(this->application->get_parameters_fluid().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      DoFs += ale_elasticity_operator->get_number_of_dofs();
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(this->mpi_comm);

    Utilities::MPI::MinMaxAvg total_time_data =
      Utilities::MPI::min_max_avg(total_time, this->mpi_comm);
    double const total_time_avg = total_time_data.avg;

    unsigned int N_time_steps = fluid_time_integrator->get_number_of_time_steps();

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
    communicator_ale->read();
  }

  void
  coupling_structure_to_fluid() const
  {
    communicator_fluid->read();
  }

  void
  coupling_fluid_to_structure() const
  {
    VectorType stress_fluid;
    fluid_operator->initialize_vector_velocity(stress_fluid);
    // calculate fluid stress at fluid-structure interface
    fluid_operator->interpolate_stress_bc(stress_fluid,
                                          fluid_time_integrator->get_velocity_np(),
                                          fluid_time_integrator->get_pressure_np());
    stress_fluid *= -1.0;

    this->precice->advance(fluid_time_integrator->get_time_step_size());
  }

  void
  solve_ale() const
  {
    bool const print_solver_info = fluid_time_integrator->print_solver_info();
    fluid_grid_motion->update(fluid_time_integrator->get_next_time(),
                              print_solver_info and not(this->is_test));

    std::shared_ptr<dealii::Mapping<dim> const> mapping =
      get_dynamic_mapping<dim, Number>(this->application->get_grid_fluid(), fluid_grid_motion);
    fluid_matrix_free->update_mapping(*mapping);

    fluid_operator->update_after_grid_motion();

    fluid_time_integrator->ale_update();
  }

  /****************************************** FLUID *******************************************/

  // grid
  std::shared_ptr<Grid<dim>> fluid_grid;

  // moving mapping (ALE)
  std::shared_ptr<GridMotionBase<dim, Number>> fluid_grid_motion;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> fluid_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     fluid_matrix_free;

  // spatial discretization
  std::shared_ptr<IncNS::SpatialOperatorBase<dim, Number>> fluid_operator;

  // temporal discretization
  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> fluid_time_integrator;

  // Postprocessor
  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> fluid_postprocessor;

  /****************************************** FLUID *******************************************/


  /************************************ ALE - MOVING MESH *************************************/

  // use a PDE solver for moving mesh problem
  std::shared_ptr<MatrixFreeData<dim, Number>> ale_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     ale_matrix_free;

  // Poisson-type mesh motion
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> ale_poisson_operator;

  // elasticity-type mesh motion
  std::shared_ptr<Structure::Operator<dim, Number>> ale_elasticity_operator;

  /************************************ ALE - MOVING MESH *************************************/


  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> communicator_fluid;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> communicator_ale;

  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/
};

} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_ */
