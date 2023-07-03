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
#include <exadg/poisson/driver.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/throughput_parameters.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                                 comm,
                            std::shared_ptr<ApplicationBase<dim, 1, Number>> app,
                            bool const                                       is_test,
                            bool const                                       is_throughput_study)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    is_throughput_study(is_throughput_study),
    application(app),
    iterations(0),
    solve_time(0.0)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);

  poisson = std::make_shared<Solver<dim, 1, Number>>();
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up Poisson solver:" << std::endl;

  application->setup();

  poisson->setup(application, mpi_comm, is_throughput_study);

  timer_tree.insert({"Poisson", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  // initialization of vectors
  dealii::Timer timer;
  timer.restart();
  dealii::LinearAlgebra::distributed::Vector<Number> rhs;
  dealii::LinearAlgebra::distributed::Vector<Number> sol;
  poisson->pde_operator->initialize_dof_vector(rhs);
  poisson->pde_operator->initialize_dof_vector(sol);
  poisson->pde_operator->prescribe_initial_conditions(sol);
  timer_tree.insert({"Poisson", "Vector init"}, timer.wall_time());

  // postprocessing of results
  timer.restart();
  poisson->postprocessor->do_postprocessing(sol);
  timer_tree.insert({"Poisson", "Postprocessing"}, timer.wall_time());

  // calculate right-hand side
  timer.restart();
  poisson->pde_operator->rhs(rhs);
  timer_tree.insert({"Poisson", "Right-hand side"}, timer.wall_time());

  // solve linear system of equations
  timer.restart();
  iterations = poisson->pde_operator->solve(sol, rhs, 0.0 /* time */);
  solve_time += timer.wall_time();

  // postprocessing of results
  timer.restart();
  poisson->postprocessor->do_postprocessing(sol);
  timer_tree.insert({"Poisson", "Postprocessing"}, timer.wall_time());
}

template<int dim, typename Number>
SolverResult
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  double const n_10 = poisson->pde_operator->get_n10();

  dealii::types::global_dof_index const DoFs = poisson->pde_operator->get_number_of_dofs();

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  double const t_10 = iterations > 0 ? solve_time * double(n_10) / double(iterations) : solve_time;

  double const tau_10 = t_10 * (double)N_mpi_processes / DoFs;

  if(not(is_test))
  {
    this->pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

    this->pcout << "Performance results for Poisson solver:" << std::endl;

    // Iterations
    this->pcout << std::endl << "Number of iterations:" << std::endl;

    this->pcout << "  Iterations n         = " << std::fixed << iterations << std::endl
                << "  Iterations n_10      = " << std::fixed << std::setprecision(1) << n_10
                << std::endl
                << "  Convergence rate rho = " << std::fixed << std::setprecision(4)
                << poisson->pde_operator->get_average_convergence_rate() << std::endl;

    // wall times
    timer_tree.insert({"Poisson"}, total_time);

    // insert sub-tree for Krylov solver
    timer_tree.insert({"Poisson"}, poisson->pde_operator->get_timings());

    pcout << std::endl << "Timings for level 1:" << std::endl;
    timer_tree.print_level(pcout, 1);

    pcout << std::endl << "Timings for level 2:" << std::endl;
    timer_tree.print_level(pcout, 2);

    pcout << std::endl << "Timings for level 3:" << std::endl;
    timer_tree.print_level(pcout, 3);

    // Throughput of linear solver in DoFs/s per core
    print_throughput_10(pcout, DoFs, t_10, N_mpi_processes);

    // Throughput in DoFs/s per core (overall costs)
    dealii::Utilities::MPI::MinMaxAvg overall_time_data =
      dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
    double const overall_time_avg = overall_time_data.avg;
    print_throughput_steady(pcout, DoFs, overall_time_avg, N_mpi_processes);

    // computational costs in CPUh
    print_costs(pcout, overall_time_avg, N_mpi_processes);

    this->pcout << print_horizontal_line() << std::endl << std::endl;
  }

  return SolverResult(application->get_parameters().degree, DoFs, n_10, tau_10);
}

template<int dim, typename Number>
std::tuple<unsigned int, dealii::types::global_dof_index, double>
Driver<dim, Number>::apply_operator(OperatorType const & operator_type,
                                    unsigned int const   n_repetitions_inner,
                                    unsigned int const   n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  dealii::LinearAlgebra::distributed::Vector<Number> dst, src;
  poisson->pde_operator->initialize_dof_vector(src);
  poisson->pde_operator->initialize_dof_vector(dst);
  src = 1.0;

  const std::function<void(void)> operator_evaluation = [&](void) {
    if(operator_type == OperatorType::Evaluate)
      poisson->pde_operator->evaluate(dst, src, 0.0);
    else if(operator_type == OperatorType::Apply)
      poisson->pde_operator->vmult(dst, src);
    else
      AssertThrow(false, dealii::ExcMessage("not implemented."));
  };

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(operator_evaluation,
                                                            application->get_parameters().degree,
                                                            n_repetitions_inner,
                                                            n_repetitions_outer,
                                                            mpi_comm);

  // calculate throughput
  dealii::types::global_dof_index const dofs = poisson->pde_operator->get_number_of_dofs();

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

} // namespace Poisson
} // namespace ExaDG
