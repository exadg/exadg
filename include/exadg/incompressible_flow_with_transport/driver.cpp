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

#include <exadg/convection_diffusion/time_integration/create_time_integrator.h>
#include <exadg/incompressible_flow_with_transport/driver.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace FTI
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm, bool const is_test)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    use_adaptive_time_stepping(false),
    N_time_steps(0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const                            degree,
                           unsigned int const                            refine_space)
{
  Timer timer;
  timer.restart();

  print_exadg_header(pcout);
  pcout << "Setting up incompressible flow with scalar transport solver:" << std::endl;

  if(not(is_test))
  {
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }
  print_MPI_info(pcout, mpi_comm);

  application = app;

  unsigned int const n_scalars = application->get_n_scalars();

  scalar_operator.resize(n_scalars);
  scalar_postprocessor.resize(n_scalars);
  scalar_time_integrator.resize(n_scalars);

  // parameters fluid
  application->set_parameters(degree, refine_space, 0 /* unused */, 0 /* unused */);
  application->get_parameters().check(pcout);

  application->get_parameters().print(pcout, "List of parameters for fluid solver:");

  // parameters scalar
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    application->set_parameters_scalar(degree, i);
    application->get_parameters_scalar(i).check();
    AssertThrow(application->get_parameters_scalar(i).problem_type ==
                  ConvDiff::ProblemType::Unsteady,
                ExcMessage("ProblemType must be unsteady!"));

    application->get_parameters_scalar(i).print(pcout,
                                                "List of parameters for scalar quantity " +
                                                  Utilities::to_string(i) + ":");
  }

  // grid
  grid = application->create_grid();
  print_grid_info(pcout, *grid);

  if(application->get_parameters().ale_formulation) // moving mesh
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(application->get_parameters_scalar(i).ale_formulation == true,
                  ExcMessage(
                    "Parameter ale_formulation is different for fluid field and scalar field"));
    }

    AssertThrow(application->get_parameters().mesh_movement_type ==
                  IncNS::MeshMovementType::Analytical,
                ExcMessage("not implemented."));

    std::shared_ptr<Function<dim>> mesh_motion;
    mesh_motion = application->create_mesh_movement_function();
    grid_motion =
      std::make_shared<GridMotionAnalytical<dim, Number>>(grid->mapping,
                                                          application->get_parameters().degree_u,
                                                          *grid->triangulation,
                                                          mesh_motion,
                                                          application->get_parameters().start_time);

    grid->attach_grid_motion(grid_motion);
  }

  // boundary conditions
  application->set_boundary_descriptor();
  IncNS::verify_boundary_conditions<dim>(application->get_boundary_descriptor(), *grid);

  // field functions
  application->set_field_functions();

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // boundary conditions
    application->set_boundary_descriptor_scalar(i);
    verify_boundary_conditions(*application->get_boundary_descriptor_scalar(i), *grid);

    // field functions
    application->set_field_functions_scalar(i);
  }


  // initialize fluid_operator
  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_operator = IncNS::create_operator<dim, Number>(grid,
                                                         application->get_boundary_descriptor(),
                                                         application->get_field_functions(),
                                                         application->get_parameters(),
                                                         "fluid",
                                                         mpi_comm);
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    fluid_operator =
      std::make_shared<IncNS::OperatorCoupled<dim, Number>>(grid,
                                                            application->get_boundary_descriptor(),
                                                            application->get_field_functions(),
                                                            application->get_parameters(),
                                                            "fluid",
                                                            mpi_comm);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_operator[i] =
      std::make_shared<ConvDiff::Operator<dim, Number>>(grid,
                                                        application->get_boundary_descriptor_scalar(
                                                          i),
                                                        application->get_field_functions_scalar(i),
                                                        application->get_parameters_scalar(i),
                                                        "scalar" + std::to_string(i),
                                                        mpi_comm);
  }

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(fluid_operator);
  for(unsigned int i = 0; i < n_scalars; ++i)
    matrix_free_data->append(scalar_operator[i]);

  matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  if(application->get_parameters().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);
  matrix_free->reinit(*grid->mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(
      application->get_parameters_scalar(i).use_cell_based_face_loops ==
        application->get_parameters().use_cell_based_face_loops,
      ExcMessage(
        "Parameter use_cell_based_face_loops should be the same for fluid and scalar transport."));
  }

  // setup Navier-Stokes operator
  if(application->get_parameters().boussinesq_term)
  {
    // assume that the first scalar field with index 0 is the active scalar that
    // couples to the incompressible Navier-Stokes equations
    fluid_operator->setup(matrix_free, matrix_free_data, scalar_operator[0]->get_dof_name());
  }
  else
  {
    fluid_operator->setup(matrix_free, matrix_free_data);
  }

  // setup convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_operator[i]->setup(matrix_free,
                              matrix_free_data,
                              fluid_operator->get_dof_name_velocity());
  }

  // setup postprocessor
  fluid_postprocessor = application->create_postprocessor();
  fluid_postprocessor->setup(*fluid_operator);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_postprocessor[i] = application->create_postprocessor_scalar(i);
    scalar_postprocessor[i]->setup(*scalar_operator[i], *grid->get_dynamic_mapping());
  }

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator = IncNS::create_time_integrator<dim, Number>(
      fluid_operator, application->get_parameters(), mpi_comm, is_test, fluid_postprocessor);
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    std::shared_ptr<IncNS::OperatorCoupled<dim, Number>> fluid_operator_coupled =
      std::dynamic_pointer_cast<IncNS::OperatorCoupled<dim, Number>>(fluid_operator);

    // initialize driver for steady state problem that depends on fluid_operator
    fluid_driver_steady = std::make_shared<DriverSteady>(fluid_operator_coupled,
                                                         application->get_parameters(),
                                                         mpi_comm,
                                                         is_test,
                                                         fluid_postprocessor);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator->setup(application->get_parameters().restarted_simulation);

    // setup solvers once time integrator has been initialized
    fluid_operator->setup_solvers(fluid_time_integrator->get_scaling_factor_time_derivative_term(),
                                  fluid_time_integrator->get_velocity());
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    fluid_driver_steady->setup();

    fluid_operator->setup_solvers(1.0 /* dummy */, fluid_driver_steady->get_velocity());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // initialize time integrator
    scalar_time_integrator[i] =
      ConvDiff::create_time_integrator<dim, Number>(scalar_operator[i],
                                                    application->get_parameters_scalar(i),
                                                    mpi_comm,
                                                    is_test,
                                                    scalar_postprocessor[i]);

    if(application->get_parameters_scalar(i).restarted_simulation == false)
    {
      // The parameter start_with_low_order has to be true.
      // This is due to the fact that the setup function of the time integrator initializes
      // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
      // start_with_low_order == false. However, the combined time step size
      // is not known at this point since we have to first communicate the time step size
      // in order to find the minimum time step size. Overwriting the time step size would
      // imply that the time step sizes are non constant for a time integrator, but the
      // time integration constants are initialized based on the assumption of constant
      // time step size. Hence, the easiest way to avoid these kind of
      // inconsistencies is to preclude the case start_with_low_order == false.
      // In case of a restart, start_with_low_order = false is possible since it has been
      // enforced that all previous time steps are identical for fluid and scalar transport.
      AssertThrow(application->get_parameters().start_with_low_order == true &&
                    application->get_parameters_scalar(i).start_with_low_order == true,
                  ExcMessage("start_with_low_order has to be true for this solver."));
    }

    scalar_time_integrator[i]->setup(application->get_parameters_scalar(i).restarted_simulation);

    // adaptive time stepping
    if(application->get_parameters().adaptive_time_stepping == true)
    {
      AssertThrow(
        application->get_parameters_scalar(i).adaptive_time_stepping == true,
        ExcMessage(
          "Adaptive time stepping has to be used for both fluid and scalar transport solvers."));

      use_adaptive_time_stepping = true;
    }
  }

  // setup solvers in case of BDF time integration (solution of linear systems of equations)
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(application->get_parameters_scalar(i).analytical_velocity_field == false,
                ExcMessage(
                  "An analytical velocity field can not be used for this coupled solver."));

    if(application->get_parameters_scalar(i).temporal_discretization ==
       ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> scalar_time_integrator_BDF =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      double const scaling_factor =
        scalar_time_integrator_BDF->get_scaling_factor_time_derivative_term();

      LinearAlgebra::distributed::Vector<Number> vector;
      fluid_operator->initialize_vector_velocity(vector);
      LinearAlgebra::distributed::Vector<Number> const * velocity = &vector;

      scalar_operator[i]->setup_solver(scaling_factor, velocity);
    }
    else
    {
      AssertThrow(application->get_parameters_scalar(i).temporal_discretization ==
                    ConvDiff::TemporalDiscretization::ExplRK,
                  ExcMessage("Not implemented."));
    }
  }

  // Boussinesq term
  // assume that the first scalar quantity with index 0 is the active scalar coupled to
  // the incompressible Navier-Stokes equations via the Boussinesq term
  if(application->get_parameters().boussinesq_term)
    scalar_operator[0]->initialize_dof_vector(temperature);

  timer_tree.insert({"Flow + transport", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  double time = std::numeric_limits<double>::max();

  // Setup time integrator and get time step size
  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    time = std::min(time, fluid_time_integrator->get_time());

  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    time = std::min(time, scalar_time_integrator[i]->get_time());
  }

  // Set the same start time for both solvers

  // fluid
  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    fluid_time_integrator->reset_time(time);

  // scalar transport
  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    scalar_time_integrator[i]->reset_time(time);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  double time_step_size = std::numeric_limits<double>::max();

  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    // Setup time integrator and get time step size
    double time_step_size_fluid = std::numeric_limits<double>::max();

    IncNS::Parameters const & fluid_param = application->get_parameters();

    // fluid
    if(fluid_time_integrator->get_time() > fluid_param.start_time - EPSILON)
      time_step_size_fluid = fluid_time_integrator->get_time_step_size();

    if(use_adaptive_time_stepping == false)
    {
      // decrease time_step in order to exactly hit end_time
      time_step_size_fluid =
        (fluid_param.end_time - fluid_param.start_time) /
        (1 + int((fluid_param.end_time - fluid_param.start_time) / time_step_size_fluid));
    }

    time_step_size = std::min(time_step_size, time_step_size_fluid);
  }

  // scalar transport
  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    ConvDiff::Parameters const & scalar_param_i = application->get_parameters_scalar(i);

    double time_step_size_scalar = std::numeric_limits<double>::max();
    if(scalar_time_integrator[i]->get_time() > scalar_param_i.start_time - EPSILON)
      time_step_size_scalar = scalar_time_integrator[i]->get_time_step_size();

    if(use_adaptive_time_stepping == false)
    {
      // decrease time_step in order to exactly hit end_time
      time_step_size_scalar =
        (scalar_param_i.end_time - scalar_param_i.start_time) /
        (1 + int((scalar_param_i.end_time - scalar_param_i.start_time) / time_step_size_scalar));
    }

    time_step_size = std::min(time_step_size, time_step_size_scalar);
  }

  if(use_adaptive_time_stepping == false)
  {
    pcout << std::endl << "Combined time step size dt = " << time_step_size << std::endl;
  }

  // Set the same time step size for both solvers

  // fluid
  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator->set_current_time_step_size(time_step_size);
  }

  // scalar transport
  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    scalar_time_integrator[i]->set_current_time_step_size(time_step_size);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_scalar_to_fluid() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // scalar transport solver (scalar 0 by definition) for the temperature and hand it over to the
  // fluid solver.
  if(application->get_parameters().boussinesq_term)
  {
    // assume that the first scalar quantity with index 0 is the active scalar coupled to
    // the incompressible Navier-Stokes equations via the Boussinesq term
    if(application->get_parameters_scalar(0).temporal_discretization ==
       ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else if(application->get_parameters_scalar(0).temporal_discretization ==
            ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    fluid_operator->set_temperature(temperature);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_fluid_to_all_scalars() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // fluid solver for the velocity field and hand it over to all scalar transport solvers.
  std::vector<LinearAlgebra::distributed::Vector<Number> const *> velocities;
  std::vector<double>                                             times;

  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator->get_velocities_and_times_np(velocities, times);
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    velocities.resize(1);
    times.resize(1);

    velocities.at(0) = &fluid_driver_steady->get_velocity();
    AssertThrow(scalar_time_integrator[0].get() != nullptr, ExcMessage("Not implemented."));
    times.at(0) = scalar_time_integrator[0]->get_next_time();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    if(application->get_parameters_scalar(i).temporal_discretization ==
       ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else if(application->get_parameters_scalar(i).temporal_discretization ==
            ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  Timer timer;
  timer.restart();

  Timer sub_timer;

  sub_timer.restart();
  grid_motion->update(fluid_time_integrator->get_next_time(), false);
  timer_tree.insert({"Flow + transport", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  matrix_free->update_mapping(*grid->get_dynamic_mapping());
  timer_tree.insert({"Flow + transport", "ALE", "Update matrix-free"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_operator->update_after_grid_motion();
  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
    scalar_operator[i]->update_after_grid_motion();
  timer_tree.insert({"Flow + transport", "ALE", "Update all operators"}, sub_timer.wall_time());

  sub_timer.restart();
  fluid_time_integrator->ale_update();
  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_bdf =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
    time_int_bdf->ale_update();
  }
  timer_tree.insert({"Flow + transport", "ALE", "Update all time integrators"},
                    sub_timer.wall_time());

  timer_tree.insert({"Flow + transport", "ALE"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // time loop
  bool finished = false;
  while(not finished)
  {
    /*
     * pre solve
     */
    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      fluid_time_integrator->advance_one_timestep_pre_solve(true);

    for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
      scalar_time_integrator[i]->advance_one_timestep_pre_solve(false);

    /*
     * ALE: move the mesh and update dependent data structures
     */
    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      if(application->get_parameters().ale_formulation) // moving mesh
        ale_update();

    /*
     *  solve
     */

    // Communicate scalar -> fluid
    communicate_scalar_to_fluid();

    // fluid: advance one time step
    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    {
      fluid_time_integrator->advance_one_timestep_solve();
    }
    else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
    {
      fluid_driver_steady->solve_steady_problem(scalar_time_integrator[0]->get_next_time(),
                                                true /*unsteady*/);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    // Communicate fluid -> all scalars
    communicate_fluid_to_all_scalars();

    // scalar transport: advance one time step
    for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
      scalar_time_integrator[i]->advance_one_timestep_solve();

    /*
     * post solve
     */
    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      fluid_time_integrator->advance_one_timestep_post_solve();

    for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
      scalar_time_integrator[i]->advance_one_timestep_post_solve();

    // Both solvers have already calculated the new, adaptive time step size individually in
    // function advance_one_timestep(). Here, we have to synchronize the time step size.
    if(use_adaptive_time_stepping == true)
      synchronize_time_step_size();

    // check if all finished
    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      finished = fluid_time_integrator->finished();
    else
      finished = true;

    for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
      finished = finished && scalar_time_integrator[i]->finished();

    ++N_time_steps;
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  pcout << "Performance results for coupled flow-transport solver:" << std::endl;

  // Iterations
  this->pcout << std::endl << "Average number of iterations:" << std::endl;

  // Fluid
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    this->fluid_time_integrator->print_iterations();
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    fluid_driver_steady->print_iterations();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // Scalar
  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // only relevant for BDF time integrator
    if(application->get_parameters_scalar(i).temporal_discretization ==
       ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_integrator_bdf->print_iterations();
    }
    else if(application->get_parameters_scalar(i).temporal_discretization ==
            ConvDiff::TemporalDiscretization::ExplRK)
    {
      this->pcout << "  Explicit solver (no systems of equations have to be solved)" << std::endl;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // Wall times

  timer_tree.insert({"Flow + transport"}, total_time);

  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    timer_tree.insert({"Flow + transport"}, fluid_time_integrator->get_timings(), "Timeloop fluid");
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    timer_tree.insert({"Flow + transport"}, fluid_driver_steady->get_timings(), "Timeloop fluid");
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    timer_tree.insert({"Flow + transport"},
                      scalar_time_integrator[i]->get_timings(),
                      "Timeloop scalar " + std::to_string(i));
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  types::global_dof_index DoFs = this->fluid_operator->get_number_of_dofs();

  for(unsigned int i = 0; i < application->get_n_scalars(); ++i)
  {
    DoFs += this->scalar_operator[i]->get_number_of_dofs();
  }

  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FTI
} // namespace ExaDG
