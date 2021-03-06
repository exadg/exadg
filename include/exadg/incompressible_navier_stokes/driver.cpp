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
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/incompressible_navier_stokes/driver.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/throughput_parameters.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm, bool const is_test)
  : mpi_comm(comm), pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0), is_test(is_test)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const                            degree,
                           unsigned int const                            refine_space,
                           unsigned int const                            refine_time,
                           bool const                                    is_throughput_study)
{
  Timer timer;
  timer.restart();

  print_exadg_header(pcout);
  pcout << "Setting up incompressible Navier-Stokes solver:" << std::endl;

  if(not(is_test))
  {
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);
  param.check_input_parameters(pcout);
  param.print(pcout, "List of input parameters:");

  // grid
  GridData grid_data;
  grid_data.triangulation_type = param.triangulation_type;
  grid_data.n_refine_global    = refine_space;
  grid_data.mapping_degree     = get_mapping_degree(param.mapping, degree);

  grid = application->create_grid(grid_data, mpi_comm);
  print_grid_info(pcout, *grid);

  if(param.ale_formulation) // moving mesh
  {
    if(param.mesh_movement_type == MeshMovementType::Analytical)
    {
      std::shared_ptr<Function<dim>> mesh_motion = application->set_mesh_movement_function();

      grid_motion = std::make_shared<GridMotionAnalytical<dim, Number>>(grid->mapping,
                                                                        grid_data.mapping_degree,
                                                                        *grid->triangulation,
                                                                        mesh_motion,
                                                                        param.start_time);
    }
    else if(param.mesh_movement_type == MeshMovementType::Poisson)
    {
      application->set_input_parameters_poisson(poisson_param);
      poisson_param.check_input_parameters();
      poisson_param.print(pcout, "List of input parameters for Poisson solver (moving mesh):");

      poisson_boundary_descriptor = std::make_shared<Poisson::BoundaryDescriptor<1, dim>>();
      application->set_boundary_conditions_poisson(poisson_boundary_descriptor);
      verify_boundary_conditions(*poisson_boundary_descriptor, *grid);

      poisson_field_functions = std::make_shared<Poisson::FieldFunctions<dim>>();
      application->set_field_functions_poisson(poisson_field_functions);

      AssertThrow(poisson_param.right_hand_side == false,
                  ExcMessage("Poisson problem is used for mesh movement. Hence, "
                             "the right-hand side has to be zero for the Poisson problem."));

      AssertThrow(poisson_param.mapping == param.mapping,
                  ExcMessage("Use the same mapping degree for Poisson mesh motion problem "
                             "as for actual application problem."));

      // initialize Poisson operator
      poisson_operator =
        std::make_shared<Poisson::Operator<dim, Number, dim>>(grid,
                                                              grid_data.mapping_degree,
                                                              poisson_boundary_descriptor,
                                                              poisson_field_functions,
                                                              poisson_param,
                                                              "Poisson",
                                                              mpi_comm);

      // initialize matrix_free
      poisson_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
      poisson_matrix_free_data->append(poisson_operator);

      poisson_matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      if(poisson_param.enable_cell_based_face_loops)
        Categorization::do_cell_based_loops(*grid->triangulation, poisson_matrix_free_data->data);
      poisson_matrix_free->reinit(*grid->mapping,
                                  poisson_matrix_free_data->get_dof_handler_vector(),
                                  poisson_matrix_free_data->get_constraint_vector(),
                                  poisson_matrix_free_data->get_quadrature_vector(),
                                  poisson_matrix_free_data->data);

      poisson_operator->setup(poisson_matrix_free, poisson_matrix_free_data);
      poisson_operator->setup_solver();

      grid_motion =
        std::make_shared<GridMotionPoisson<dim, Number>>(grid->mapping, poisson_operator);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    grid->attach_grid_motion(grid_motion);
  }

  // boundary conditions
  boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
  application->set_boundary_conditions(boundary_descriptor);
  IncNS::verify_boundary_conditions<dim>(boundary_descriptor, *grid);

  // field functions
  field_functions = std::make_shared<FieldFunctions<dim>>();
  application->set_field_functions(field_functions);

  if(param.solver_type == SolverType::Unsteady)
  {
    pde_operator = create_operator<dim, Number>(
      grid, degree, boundary_descriptor, field_functions, param, "fluid", mpi_comm);
  }
  else if(param.solver_type == SolverType::Steady)
  {
    pde_operator = std::make_shared<IncNS::OperatorCoupled<dim, Number>>(
      grid, degree, boundary_descriptor, field_functions, param, "fluid", mpi_comm);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  if(param.use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);
  matrix_free->reinit(*grid->get_dynamic_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  // setup Navier-Stokes operator
  pde_operator->setup(matrix_free, matrix_free_data);

  if(!is_throughput_study)
  {
    // setup postprocessor
    postprocessor = application->create_postprocessor(degree, mpi_comm);
    postprocessor->setup(*pde_operator);

    // setup time integrator before calling setup_solvers
    // (this is necessary since the setup of the solvers
    // depends on quantities such as the time_step_size or gamma0!!!)
    if(param.solver_type == SolverType::Unsteady)
    {
      time_integrator = create_time_integrator<dim, Number>(
        pde_operator, param, refine_time, mpi_comm, is_test, postprocessor);
    }
    else if(param.solver_type == SolverType::Steady)
    {
      std::shared_ptr<OperatorCoupled<dim, Number>> operator_coupled =
        std::dynamic_pointer_cast<OperatorCoupled<dim, Number>>(pde_operator);

      // initialize driver for steady state problem that depends on pde_operator
      driver_steady = std::make_shared<DriverSteadyProblems<dim, Number>>(
        operator_coupled, param, mpi_comm, is_test, postprocessor);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    if(param.solver_type == SolverType::Unsteady)
    {
      time_integrator->setup(param.restarted_simulation);

      pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
                                  time_integrator->get_velocity());
    }
    else if(param.solver_type == SolverType::Steady)
    {
      driver_steady->setup();

      pde_operator->setup_solvers(1.0 /* dummy */, driver_steady->get_velocity());
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  timer_tree.insert({"Incompressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  // move the mesh and update dependent data structures
  Timer timer;
  timer.restart();

  Timer sub_timer;

  sub_timer.restart();
  grid_motion->update(time_integrator->get_next_time(), false);
  timer_tree.insert({"Incompressible flow", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  matrix_free->update_mapping(*grid->get_dynamic_mapping());
  timer_tree.insert({"Incompressible flow", "ALE", "Update matrix-free"}, sub_timer.wall_time());

  sub_timer.restart();
  pde_operator->update_after_grid_motion();
  timer_tree.insert({"Incompressible flow", "ALE", "Update operator"}, sub_timer.wall_time());

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
  if(param.problem_type == ProblemType::Unsteady)
  {
    // stability analysis (uncomment if desired)
    // time_integrator->postprocessing_stability_analysis();

    if(param.ale_formulation == true)
    {
      while(not time_integrator->finished())
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
  else if(param.problem_type == ProblemType::Steady)
  {
    if(param.solver_type == SolverType::Unsteady)
    {
      time_integrator->timeloop_steady_problem();
    }
    else if(param.solver_type == SolverType::Steady)
    {
      // solve steady problem
      driver_steady->solve_steady_problem();
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
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
  if(param.solver_type == SolverType::Unsteady)
  {
    pcout << std::endl << "Average number of iterations:" << std::endl;

    time_integrator->print_iterations();
  }

  // Wall times
  timer_tree.insert({"Incompressible flow"}, total_time);

  if(param.solver_type == SolverType::Unsteady)
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
  types::global_dof_index const DoFs            = pde_operator->get_number_of_dofs();
  unsigned int const            N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  if(param.solver_type == SolverType::Unsteady)
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
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(unsigned int const  degree,
                                    std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  (void)degree;

  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  OperatorType operator_type;
  string_to_enum(operator_type, operator_type_string);

  AssertThrow(param.degree_p == DegreePressure::MixedOrder,
              ExcMessage(
                "The function get_dofs_per_element() assumes mixed-order polynomials for "
                "velocity and pressure. Additional operator types have to be introduced to "
                "enable equal-order polynomials for this throughput study."));

  // check that the operator type is consistent with the solution approach (coupled vs. splitting)
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    AssertThrow(operator_type == OperatorType::ConvectiveOperator ||
                  operator_type == OperatorType::CoupledNonlinearResidual ||
                  operator_type == OperatorType::CoupledLinearized ||
                  operator_type == OperatorType::InverseMassOperator,
                ExcMessage("Invalid operator specified for coupled solution approach."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    AssertThrow(operator_type == OperatorType::ConvectiveOperator ||
                  operator_type == OperatorType::PressurePoissonOperator ||
                  operator_type == OperatorType::HelmholtzOperator ||
                  operator_type == OperatorType::ProjectionOperator ||
                  operator_type == OperatorType::InverseMassOperator,
                ExcMessage("Invalid operator specified for dual splitting scheme."));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    AssertThrow(operator_type == OperatorType::ConvectiveOperator ||
                  operator_type == OperatorType::PressurePoissonOperator ||
                  operator_type == OperatorType::VelocityConvDiffOperator ||
                  operator_type == OperatorType::ProjectionOperator ||
                  operator_type == OperatorType::InverseMassOperator,
                ExcMessage("Invalid operator specified for pressure-correction scheme."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // Vectors needed for coupled solution approach
  LinearAlgebra::distributed::BlockVector<Number> dst1, src1;

  // ... for dual splitting, pressure-correction.
  LinearAlgebra::distributed::Vector<Number> dst2, src2;

  // set velocity required for evaluation of linearized operators
  LinearAlgebra::distributed::Vector<Number> velocity;
  pde_operator->initialize_vector_velocity(velocity);
  velocity = 1.0;
  pde_operator->set_velocity_ptr(velocity);

  // initialize vectors
  if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    pde_operator->initialize_block_vector_velocity_pressure(dst1);
    pde_operator->initialize_block_vector_velocity_pressure(src1);
    src1 = 1.0;

    if(operator_type == OperatorType::ConvectiveOperator ||
       operator_type == OperatorType::InverseMassOperator)
    {
      pde_operator->initialize_vector_velocity(src2);
      pde_operator->initialize_vector_velocity(dst2);
    }
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    if(operator_type == OperatorType::ConvectiveOperator ||
       operator_type == OperatorType::HelmholtzOperator ||
       operator_type == OperatorType::ProjectionOperator ||
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
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    if(operator_type == OperatorType::VelocityConvDiffOperator ||
       operator_type == OperatorType::ProjectionOperator ||
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
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    src2 = 1.0;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  const std::function<void(void)> operator_evaluation = [&](void) {
    // clang-format off
    if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
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
        AssertThrow(false,ExcMessage("Not implemented."));
    }
    else if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
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
        AssertThrow(false,ExcMessage("Not implemented."));
    }
    else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
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
        AssertThrow(false,ExcMessage("Not implemented."));
    }
    else
    {
      AssertThrow(false,ExcMessage("Not implemented."));
    }
    // clang-format on

    return;
  };

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(
    operator_evaluation, degree, n_repetitions_inner, n_repetitions_outer, mpi_comm);

  // calculate throughput
  types::global_dof_index dofs      = 0;
  unsigned int            fe_degree = 1;

  if(operator_type == OperatorType::CoupledNonlinearResidual ||
     operator_type == OperatorType::CoupledLinearized)
  {
    dofs = pde_operator->get_dof_handler_u().n_dofs() + pde_operator->get_dof_handler_p().n_dofs();

    fe_degree = pde_operator->get_polynomial_degree();
  }
  else if(operator_type == OperatorType::ConvectiveOperator ||
          operator_type == OperatorType::VelocityConvDiffOperator ||
          operator_type == OperatorType::HelmholtzOperator ||
          operator_type == OperatorType::ProjectionOperator ||
          operator_type == OperatorType::InverseMassOperator)
  {
    dofs = pde_operator->get_dof_handler_u().n_dofs();

    fe_degree = pde_operator->get_polynomial_degree();
  }
  else if(operator_type == OperatorType::PressurePoissonOperator)
  {
    dofs = pde_operator->get_dof_handler_p().n_dofs();

    fe_degree = param.get_degree_p(pde_operator->get_polynomial_degree());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double const throughput = (double)dofs / wall_time;

  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

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

  return std::tuple<unsigned int, types::global_dof_index, double>(fe_degree, dofs, throughput);
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace IncNS
} // namespace ExaDG
