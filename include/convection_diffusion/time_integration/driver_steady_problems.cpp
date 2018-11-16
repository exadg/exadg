/*
 * driver_steady_problems.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "driver_steady_problems.h"

#include "../interface_space_time/operator.h"

namespace ConvDiff
{
template<typename Number>
DriverSteadyProblems<Number>::DriverSteadyProblems(std::shared_ptr<Operator> operator_in,
                                                   InputParameters const &   param_in)
  : pde_operator(operator_in), param(param_in), total_time(0.0)
{
}

template<typename Number>
void
DriverSteadyProblems<Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<typename Number>
void
DriverSteadyProblems<Number>::solve_problem()
{
  global_timer.restart();

  postprocessing();

  solve();

  postprocessing();

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

template<typename Number>
void
DriverSteadyProblems<Number>::analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max "
        << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  pcout << "  Global time:         " << std::scientific << std::setprecision(4) << std::setw(10)
        << data.min << " " << std::setprecision(4) << std::setw(10) << data.avg << " "
        << std::setprecision(4) << std::setw(10) << data.max << " "
        << "          "
        << "  " << std::setw(6) << std::left << data.min_index << " " << std::setw(6) << std::left
        << data.max_index << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template<typename Number>
void
DriverSteadyProblems<Number>::initialize_vectors()
{
  // solution
  pde_operator->initialize_dof_vector(solution);

  // rhs_vector
  pde_operator->initialize_dof_vector(rhs_vector);
}

template<typename Number>
void
DriverSteadyProblems<Number>::initialize_solution()
{
  double time = 0.0;
  pde_operator->prescribe_initial_conditions(solution, time);
}

template<typename Number>
void
DriverSteadyProblems<Number>::solve()
{
  Timer timer;
  timer.restart();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Solving steady state problem ..." << std::endl;

  // calculate rhs vector
  pde_operator->rhs(rhs_vector);

  // solve linear system of equations
  unsigned int iterations = pde_operator->solve(solution, rhs_vector);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl
              << "Solve linear system of equations:" << std::endl
              << "  Iterations: " << std::setw(6) << std::right << iterations
              << "\t Wall time [s]: " << std::scientific << std::setprecision(4)
              << timer.wall_time() << std::endl;
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "... done!" << std::endl;
}

template<typename Number>
void
DriverSteadyProblems<Number>::postprocessing() const
{
  pde_operator->do_postprocessing(solution);
}

// instantiate for float and double
template class DriverSteadyProblems<float>;
template class DriverSteadyProblems<double>;

} // namespace ConvDiff
