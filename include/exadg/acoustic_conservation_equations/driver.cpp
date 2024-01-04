/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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
#include <exadg/acoustic_conservation_equations/driver.h>
#include <exadg/operators/throughput_parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Acoustics
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

  pcout << std::endl << "Setting up acoustic conservation equations solver:" << std::endl;

  application->setup(grid, mapping);

  pde_operator =
    std::make_shared<SpatialOperator<dim, Number>>(grid,
                                                   mapping,
                                                   application->get_boundary_descriptor(),
                                                   application->get_field_functions(),
                                                   application->get_parameters(),
                                                   "acoustic",
                                                   mpi_comm);

  // setup PDE operator
  pde_operator->setup();

  if(not is_throughput_study)
  {
    // setup postprocessor
    postprocessor = application->create_postprocessor();
    postprocessor->setup(*pde_operator);

    // create and setup time integrator
    time_integrator = std::make_shared<TimeIntAdamsBashforthMoulton<Number>>(
      pde_operator, application->get_parameters(), postprocessor, mpi_comm, is_test);
    time_integrator->setup(application->get_parameters().restarted_simulation);
  }

  timer_tree.insert({"Acoustic conservation equations", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  time_integrator->timeloop();
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl << print_horizontal_line() << std::endl << std::endl;

  pcout << "Performance results for acoustic conservation equations solver:" << std::endl;

  // Iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;
  time_integrator->print_iterations();

  // Wall times
  timer_tree.insert({"Acoustic conservation equations"}, total_time);

  timer_tree.insert({"Acoustic conservation equations"}, time_integrator->get_timings());

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

  unsigned int const N_time_steps = time_integrator->get_number_of_time_steps();
  print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  pcout << print_horizontal_line() << std::endl << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, dealii::types::global_dof_index, double>
Driver<dim, Number>::apply_operator(OperatorType const & operator_type,
                                    unsigned int const   n_repetitions_inner,
                                    unsigned int const   n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  // Vectors needed for coupled solution approach
  dealii::LinearAlgebra::distributed::BlockVector<Number> dst, src;

  // initialize vectors
  pde_operator->initialize_dof_vector(dst);
  pde_operator->initialize_dof_vector(src);
  src = 1.0;

  // evaluate operator
  const std::function<void(void)> operator_evaluation = [&](void) {
    if(operator_type == OperatorType::AcousticOperator)
      pde_operator->evaluate_acoustic_operator(dst, src, 0.0);
    else if(operator_type == OperatorType::ScaledInverseMassOperator)
      pde_operator->apply_scaled_inverse_mass_operator(dst, src);
    else
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
  };

  // calculate throughput

  // determine DoFs and degree
  dealii::types::global_dof_index const dofs      = pde_operator->get_number_of_dofs();
  unsigned int const                    fe_degree = application->get_parameters().degree_p;

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

} // namespace Acoustics
} // namespace ExaDG
