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
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/operators/throughput_parameters.h>
#include <exadg/structure/driver.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Structure
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

  pcout << std::endl << "Setting up elasticity solver:" << std::endl;

  application->setup();

  // setup spatial operator
  pde_operator = std::make_shared<Operator<dim, Number>>(application->get_grid(),
                                                         application->get_mapping(),
                                                         application->get_boundary_descriptor(),
                                                         application->get_field_functions(),
                                                         application->get_material_descriptor(),
                                                         application->get_parameters(),
                                                         "elasticity",
                                                         mpi_comm);

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  matrix_free->reinit(*application->get_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  pde_operator->setup(matrix_free, matrix_free_data);

  if(not is_throughput_study)
  {
    // initialize postprocessor
    postprocessor = application->create_postprocessor();
    postprocessor->setup(pde_operator->get_dof_handler(), pde_operator->get_mapping());

    // initialize time integrator/driver
    if(application->get_parameters().problem_type == ProblemType::Unsteady)
    {
      time_integrator = std::make_shared<TimeIntGenAlpha<dim, Number>>(
        pde_operator, postprocessor, application->get_parameters(), mpi_comm, is_test);
      time_integrator->setup(application->get_parameters().restarted_simulation);
    }
    else if(application->get_parameters().problem_type == ProblemType::Steady)
    {
      driver_steady = std::make_shared<DriverSteady<dim, Number>>(
        pde_operator, postprocessor, application->get_parameters(), mpi_comm, is_test);
      driver_steady->setup();
    }
    else if(application->get_parameters().problem_type == ProblemType::QuasiStatic)
    {
      driver_quasi_static = std::make_shared<DriverQuasiStatic<dim, Number>>(
        pde_operator, postprocessor, application->get_parameters(), mpi_comm, is_test);
      driver_quasi_static->setup();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    if(application->get_parameters().problem_type == ProblemType::Unsteady)
      pde_operator->setup_solver(time_integrator->get_scaling_factor_mass());
    else
      pde_operator->setup_solver(0.0);
  }

  timer_tree.insert({"Elasticity", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    time_integrator->compute_initial_acceleration(
      application->get_parameters().restarted_simulation);
    time_integrator->timeloop();
  }
  else if(application->get_parameters().problem_type == ProblemType::Steady)
  {
    driver_steady->solve();
  }
  else if(application->get_parameters().problem_type == ProblemType::QuasiStatic)
  {
    driver_quasi_static->solve();
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

  pcout << "Performance results for elasticity solver:" << std::endl;

  // Iterations
  if(application->get_parameters().problem_type == ProblemType::QuasiStatic)
  {
    pcout << std::endl << "Average number of iterations:" << std::endl;
    driver_quasi_static->print_iterations();
  }
  else if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    pcout << std::endl << "Average number of iterations:" << std::endl;
    time_integrator->print_iterations();
  }

  timer_tree.insert({"Elasticity"}, total_time);

  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    timer_tree.insert({"Elasticity"}, time_integrator->get_timings());
  }
  else if(application->get_parameters().problem_type == ProblemType::Steady)
  {
    timer_tree.insert({"Elasticity"}, driver_steady->get_timings());
  }
  else if(application->get_parameters().problem_type == ProblemType::QuasiStatic)
  {
    timer_tree.insert({"Elasticity"}, driver_quasi_static->get_timings());
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index const DoFs = pde_operator->get_number_of_dofs();
  unsigned int const N_mpi_processes         = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  if(application->get_parameters().problem_type == ProblemType::Unsteady)
  {
    unsigned int const N_time_steps = time_integrator->get_number_of_time_steps();
    print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);
  }
  else
  {
    print_throughput_steady(pcout, DoFs, total_time_avg, N_mpi_processes);
  }

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, dealii::types::global_dof_index, double>
Driver<dim, Number>::apply_operator(OperatorType const & operator_type,
                                    unsigned int const   n_repetitions_inner,
                                    unsigned int const   n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  dealii::LinearAlgebra::distributed::Vector<Number> dst, src, linearization;
  pde_operator->initialize_dof_vector(src);
  pde_operator->initialize_dof_vector(dst);
  src = 1.0;

  if(application->get_parameters().large_deformation and operator_type == OperatorType::Apply)
  {
    pde_operator->initialize_dof_vector(linearization);
    linearization = 1.0;
  }

  const std::function<void(void)> operator_evaluation = [&](void) {
    if(operator_type == OperatorType::Evaluate)
    {
      pde_operator->evaluate_elasticity_operator(dst, src, 1.0, 0.0);
    }
    else if(operator_type == OperatorType::Apply)
    {
      pde_operator->apply_elasticity_operator(dst, src, linearization, 1.0, 0.0);
    }
  };

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(operator_evaluation,
                                                            application->get_parameters().degree,
                                                            n_repetitions_inner,
                                                            n_repetitions_outer,
                                                            mpi_comm);

  // calculate throughput
  dealii::types::global_dof_index const dofs = pde_operator->get_number_of_dofs();

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

  return std::tuple<unsigned int, dealii::types::global_dof_index, double>(
    application->get_parameters().degree, dofs, throughput);
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Structure
} // namespace ExaDG
