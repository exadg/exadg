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

// likwid
#ifdef EXADG_WITH_LIKWID
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/grid/get_dynamic_mapping.h>
#include <exadg/incompressible_navier_stokes/driver.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/throughput_parameters.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test,
                            bool const                                    is_throughput_study)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    is_throughput_study(is_throughput_study),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up incompressible Navier-Stokes solver:" << std::endl;

  application->setup();

  // moving mesh (ALE formulation)
  if(application->get_parameters().ale_formulation)
  {
    if(application->get_parameters().mesh_movement_type == MeshMovementType::Function)
    {
      std::shared_ptr<dealii::Function<dim>> mesh_motion =
        application->create_mesh_movement_function();

      grid_motion = std::make_shared<GridMotionFunction<dim, Number>>(
        application->get_mapping(),
        application->get_parameters().mapping_degree,
        *application->get_grid()->triangulation,
        mesh_motion,
        application->get_parameters().start_time);
    }
    else if(application->get_parameters().mesh_movement_type == MeshMovementType::Poisson)
    {
      application->setup_poisson();

      grid_motion = std::make_shared<GridMotionPoisson<dim, Number>>(
        application->get_grid(),
        application->get_mapping(),
        application->get_boundary_descriptor_poisson(),
        application->get_field_functions_poisson(),
        application->get_parameters_poisson(),
        "Poisson",
        mpi_comm);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    helpers_ale = std::make_shared<HelpersALE<Number>>();

    helpers_ale->move_grid = [&](double const & time) {
      grid_motion->update(time,
                          false /* print_solver_info */,
                          this->time_integrator->get_number_of_time_steps());
    };

    helpers_ale->update_pde_operator_after_grid_motion = [&]() {
      std::shared_ptr<dealii::Mapping<dim> const> mapping =
        get_dynamic_mapping<dim, Number>(application->get_mapping(), grid_motion);
      matrix_free->update_mapping(*mapping);

      pde_operator->update_after_grid_motion();
    };
  }

  if(application->get_parameters().solver_type == SolverType::Unsteady)
  {
    pde_operator =
      create_operator<dim, Number>(application->get_grid(),
                                   get_dynamic_mapping<dim, Number>(application->get_mapping(),
                                                                    grid_motion),
                                   application->get_boundary_descriptor(),
                                   application->get_field_functions(),
                                   application->get_parameters(),
                                   "fluid",
                                   mpi_comm);
  }
  else if(application->get_parameters().solver_type == SolverType::Steady)
  {
    pde_operator = std::make_shared<IncNS::OperatorCoupled<dim, Number>>(
      application->get_grid(),
      get_dynamic_mapping<dim, Number>(application->get_mapping(), grid_motion),
      application->get_boundary_descriptor(),
      application->get_field_functions(),
      application->get_parameters(),
      "fluid",
      mpi_comm);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  if(application->get_parameters().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*application->get_grid()->triangulation,
                                        matrix_free_data->data);
  std::shared_ptr<dealii::Mapping<dim> const> mapping =
    get_dynamic_mapping<dim, Number>(application->get_mapping(), grid_motion);
  matrix_free->reinit(*mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  // setup Navier-Stokes operator
  pde_operator->setup(matrix_free, matrix_free_data);

  if(not is_throughput_study)
  {
    // setup postprocessor
    postprocessor = application->create_postprocessor();
    postprocessor->setup(*pde_operator);

    // setup time integrator before calling setup_solvers
    // (this is necessary since the setup of the solvers
    // depends on quantities such as the time_step_size or gamma0!)
    if(application->get_parameters().solver_type == SolverType::Unsteady)
    {
      time_integrator = create_time_integrator<dim, Number>(
        pde_operator, helpers_ale, postprocessor, application->get_parameters(), mpi_comm, is_test);
    }
    else if(application->get_parameters().solver_type == SolverType::Steady)
    {
      std::shared_ptr<OperatorCoupled<dim, Number>> operator_coupled =
        std::dynamic_pointer_cast<OperatorCoupled<dim, Number>>(pde_operator);

      // initialize driver for steady state problem that depends on pde_operator
      driver_steady = std::make_shared<DriverSteadyProblems<dim, Number>>(
        operator_coupled, postprocessor, application->get_parameters(), mpi_comm, is_test);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    if(application->get_parameters().solver_type == SolverType::Unsteady)
    {
      time_integrator->setup(application->get_parameters().restarted_simulation);

      pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
                                  time_integrator->get_velocity());
    }
    else if(application->get_parameters().solver_type == SolverType::Steady)
    {
      driver_steady->setup();

      pde_operator->setup_solvers(1.0 /* dummy */, driver_steady->get_velocity());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  timer_tree.insert({"Incompressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  // move the mesh and update dependent data structures
  dealii::Timer timer;
  timer.restart();

  dealii::Timer sub_timer;

  sub_timer.restart();
  helpers_ale->move_grid(time_integrator->get_next_time());
  timer_tree.insert({"Incompressible flow", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  helpers_ale->update_pde_operator_after_grid_motion();
  timer_tree.insert({"Incompressible flow", "ALE", "Update matrix-free / PDE operator"},
                    sub_timer.wall_time());

  sub_timer.restart();
  time_integrator->ale_update();
  timer_tree.insert({"Incompressible flow", "ALE", "Update time integrator"},
                    sub_timer.wall_time());

  timer_tree.insert({"Incompressible flow", "ALE"}, timer.wall_time());
}


template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    // stability analysis (uncomment if desired)
    // time_integrator->postprocessing_stability_analysis();

    if(application->get_parameters().ale_formulation == true)
    {
      while(not(time_integrator->finished()))
      {
        time_integrator->advance_one_timestep_pre_solve(true);

        ale_update();

        time_integrator->advance_one_timestep_solve();

        time_integrator->advance_one_timestep_post_solve();
      }
    }
    else
    {
      time_integrator->timeloop();
    }
  }
  else if(application->get_parameters().problem_type == ProblemType::Steady)
  {
    if(application->get_parameters().solver_type == SolverType::Unsteady)
    {
      time_integrator->timeloop_steady_problem();
    }
    else if(application->get_parameters().solver_type == SolverType::Steady)
    {
      // solve steady problem
      driver_steady->solve();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for incompressible Navier-Stokes solver:" << std::endl;

  // Iterations
  if(application->get_parameters().solver_type == SolverType::Unsteady)
  {
    pcout << std::endl << "Average number of iterations:" << std::endl;

    time_integrator->print_iterations();
  }

  // Wall times
  timer_tree.insert({"Incompressible flow"}, total_time);

  if(application->get_parameters().solver_type == SolverType::Unsteady)
  {
    timer_tree.insert({"Incompressible flow"}, time_integrator->get_timings());
  }
  else
  {
    timer_tree.insert({"Incompressible flow"}, driver_steady->get_timings());
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index const DoFs = pde_operator->get_number_of_dofs();
  unsigned int const N_mpi_processes         = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg overall_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const overall_time_avg = overall_time_data.avg;

  if(application->get_parameters().solver_type == SolverType::Unsteady)
  {
    unsigned int const N_time_steps = time_integrator->get_number_of_time_steps();
    print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);
  }
  else
  {
    print_throughput_steady(pcout, DoFs, overall_time_avg, N_mpi_processes);
  }

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, dealii::types::global_dof_index, double>
Driver<dim, Number>::apply_operator(std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  OperatorType operator_type;
  Utilities::string_to_enum(operator_type, operator_type_string);

  AssertThrow(application->get_parameters().degree_p == DegreePressure::MixedOrder,
              dealii::ExcMessage(
                "The function get_dofs_per_element() assumes mixed-order polynomials for "
                "velocity and pressure. Additional operator types have to be introduced to "
                "enable equal-order polynomials for this throughput study."));

  // check that the operator type is consistent with the solution approach (coupled vs. splitting)
  if(application->get_parameters().temporal_discretization ==
     TemporalDiscretization::BDFCoupledSolution)
  {
    AssertThrow(operator_type == OperatorType::ConvectiveOperator or
                  operator_type == OperatorType::CoupledNonlinearResidual or
                  operator_type == OperatorType::CoupledLinearized or
                  operator_type == OperatorType::InverseMassOperator,
                dealii::ExcMessage("Invalid operator specified for coupled solution approach."));
  }
  else if(application->get_parameters().temporal_discretization ==
          TemporalDiscretization::BDFDualSplittingScheme)
  {
    AssertThrow(operator_type == OperatorType::ConvectiveOperator or
                  operator_type == OperatorType::PressurePoissonOperator or
                  operator_type == OperatorType::HelmholtzOperator or
                  operator_type == OperatorType::ProjectionOperator or
                  operator_type == OperatorType::InverseMassOperator,
                dealii::ExcMessage("Invalid operator specified for dual splitting scheme."));
  }
  else if(application->get_parameters().temporal_discretization ==
          TemporalDiscretization::BDFPressureCorrection)
  {
    AssertThrow(operator_type == OperatorType::ConvectiveOperator or
                  operator_type == OperatorType::PressurePoissonOperator or
                  operator_type == OperatorType::VelocityConvDiffOperator or
                  operator_type == OperatorType::ProjectionOperator or
                  operator_type == OperatorType::InverseMassOperator,
                dealii::ExcMessage("Invalid operator specified for pressure-correction scheme."));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  // Vectors needed for coupled solution approach
  dealii::LinearAlgebra::distributed::BlockVector<Number> dst1, src1;

  // ... for dual splitting, pressure-correction.
  dealii::LinearAlgebra::distributed::Vector<Number> dst2, src2;

  // set velocity required for evaluation of linearized operators
  dealii::LinearAlgebra::distributed::Vector<Number> velocity;
  pde_operator->initialize_vector_velocity(velocity);
  velocity = 1.0;
  pde_operator->set_velocity_ptr(velocity);

  // initialize vectors
  if(application->get_parameters().temporal_discretization ==
     TemporalDiscretization::BDFCoupledSolution)
  {
    pde_operator->initialize_block_vector_velocity_pressure(dst1);
    pde_operator->initialize_block_vector_velocity_pressure(src1);
    src1 = 1.0;

    if(operator_type == OperatorType::ConvectiveOperator or
       operator_type == OperatorType::InverseMassOperator)
    {
      pde_operator->initialize_vector_velocity(src2);
      pde_operator->initialize_vector_velocity(dst2);
    }
  }
  else if(application->get_parameters().temporal_discretization ==
          TemporalDiscretization::BDFDualSplittingScheme)
  {
    if(operator_type == OperatorType::ConvectiveOperator or
       operator_type == OperatorType::HelmholtzOperator or
       operator_type == OperatorType::ProjectionOperator or
       operator_type == OperatorType::InverseMassOperator)
    {
      pde_operator->initialize_vector_velocity(src2);
      pde_operator->initialize_vector_velocity(dst2);
    }
    else if(operator_type == OperatorType::PressurePoissonOperator)
    {
      pde_operator->initialize_vector_pressure(src2);
      pde_operator->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else if(application->get_parameters().temporal_discretization ==
          TemporalDiscretization::BDFPressureCorrection)
  {
    if(operator_type == OperatorType::VelocityConvDiffOperator or
       operator_type == OperatorType::ProjectionOperator or
       operator_type == OperatorType::InverseMassOperator)
    {
      pde_operator->initialize_vector_velocity(src2);
      pde_operator->initialize_vector_velocity(dst2);
    }
    else if(operator_type == OperatorType::PressurePoissonOperator)
    {
      pde_operator->initialize_vector_pressure(src2);
      pde_operator->initialize_vector_pressure(dst2);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  const std::function<void(void)> operator_evaluation = [&](void) {
    // clang-format off
    if(application->get_parameters().temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      std::shared_ptr<OperatorCoupled<dim, Number>> operator_coupled =
          std::dynamic_pointer_cast<OperatorCoupled<dim, Number>>(pde_operator);

      if(operator_type == OperatorType::CoupledNonlinearResidual)
        operator_coupled->evaluate_nonlinear_residual(dst1,src1,&src1.block(0), 0.0, 1.0);
      else if(operator_type == OperatorType::CoupledLinearized)
        operator_coupled->apply_linearized_problem(dst1,src1, 0.0, 1.0);
      else if(operator_type == OperatorType::ConvectiveOperator)
        operator_coupled->evaluate_convective_term(dst2,src2,0.0);
      else if(operator_type == OperatorType::InverseMassOperator)
        operator_coupled->apply_inverse_mass_operator(dst2,src2);
      else
        AssertThrow(false,dealii::ExcMessage("Not implemented."));
    }
    else if(application->get_parameters().temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      std::shared_ptr<OperatorDualSplitting<dim, Number>>      operator_dual_splitting =
          std::dynamic_pointer_cast<OperatorDualSplitting<dim, Number>>(pde_operator);

      if(operator_type == OperatorType::HelmholtzOperator)
        operator_dual_splitting->apply_helmholtz_operator(dst2,src2);
      else if(operator_type == OperatorType::ConvectiveOperator)
        operator_dual_splitting->evaluate_convective_term(dst2,src2,0.0);
      else if(operator_type == OperatorType::ProjectionOperator)
        operator_dual_splitting->apply_projection_operator(dst2,src2);
      else if(operator_type == OperatorType::PressurePoissonOperator)
        operator_dual_splitting->apply_laplace_operator(dst2,src2);
      else if(operator_type == OperatorType::InverseMassOperator)
        operator_dual_splitting->apply_inverse_mass_operator(dst2,src2);
      else
        AssertThrow(false,dealii::ExcMessage("Not implemented."));
    }
    else if(application->get_parameters().temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      std::shared_ptr<OperatorPressureCorrection<dim, Number>> operator_pressure_correction =
          std::dynamic_pointer_cast<OperatorPressureCorrection<dim, Number>>(pde_operator);

      if(operator_type == OperatorType::VelocityConvDiffOperator)
        operator_pressure_correction->apply_momentum_operator(dst2,src2);
      else if(operator_type == OperatorType::ProjectionOperator)
        operator_pressure_correction->apply_projection_operator(dst2,src2);
      else if(operator_type == OperatorType::PressurePoissonOperator)
        operator_pressure_correction->apply_laplace_operator(dst2,src2);
      else if(operator_type == OperatorType::InverseMassOperator)
        operator_pressure_correction->apply_inverse_mass_operator(dst2,src2);
      else
        AssertThrow(false,dealii::ExcMessage("Not implemented."));
    }
    else
    {
      AssertThrow(false,dealii::ExcMessage("Not implemented."));
    }
    // clang-format on

    return;
  };

  // calculate throughput

  // determine DoFs and degree
  dealii::types::global_dof_index dofs      = 0;
  unsigned int                    fe_degree = 1;

  if(operator_type == OperatorType::CoupledNonlinearResidual or
     operator_type == OperatorType::CoupledLinearized)
  {
    dofs = pde_operator->get_dof_handler_u().n_dofs() + pde_operator->get_dof_handler_p().n_dofs();

    fe_degree = application->get_parameters().degree_u;
  }
  else if(operator_type == OperatorType::ConvectiveOperator or
          operator_type == OperatorType::VelocityConvDiffOperator or
          operator_type == OperatorType::HelmholtzOperator or
          operator_type == OperatorType::ProjectionOperator or
          operator_type == OperatorType::InverseMassOperator)
  {
    dofs = pde_operator->get_dof_handler_u().n_dofs();

    fe_degree = application->get_parameters().degree_u;
  }
  else if(operator_type == OperatorType::PressurePoissonOperator)
  {
    dofs = pde_operator->get_dof_handler_p().n_dofs();

    fe_degree = application->get_parameters().get_degree_p(application->get_parameters().degree_u);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(
    operator_evaluation, fe_degree, n_repetitions_inner, n_repetitions_outer, mpi_comm);

  double const throughput = (double)dofs / wall_time;

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  if(not(is_test))
  {
    // clang-format off
    pcout << std::endl
          << std::scientific << std::setprecision(4)
          << "DoFs/sec:        " << throughput << std::endl
          << "DoFs/(sec*core): " << throughput/(double)N_mpi_processes << std::endl;
    // clang-format on
  }

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, dealii::types::global_dof_index, double>(fe_degree,
                                                                           dofs,
                                                                           throughput);
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace IncNS
} // namespace ExaDG
