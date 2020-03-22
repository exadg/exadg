/*
 * driver_steady_problems.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "driver_steady_problems.h"

namespace Structure
{
template<int dim, typename Number>
DriverSteady<dim, Number>::DriverSteady(
  std::shared_ptr<Operator<dim, Number>>      operator_in,
  std::shared_ptr<PostProcessor<dim, Number>> postprocessor_in,
  InputParameters const &                     param_in,
  MPI_Comm const &                            mpi_comm_in)
  : pde_operator(operator_in),
    postprocessor(postprocessor_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    computing_times(1)
{
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::solve_problem()
{
  postprocessing();

  solve();

  postprocessing();
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::initialize_vectors()
{
  // solution
  pde_operator->initialize_dof_vector(solution);

  // rhs_vector
  pde_operator->initialize_dof_vector(rhs_vector);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::initialize_solution()
{
  double time = 0.0;
  pde_operator->prescribe_initial_conditions(solution, time);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::solve()
{
  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  Timer timer;
  timer.restart();

  // calculate rhs vector
  pde_operator->rhs(rhs_vector);

  // solve system of equations
  unsigned int iterations =
    pde_operator->solve(solution, rhs_vector, /* update_preconditioner = */ false);

  computing_times[0] += timer.wall_time();

  // write output
  pcout << std::endl
        << "Solve linear system of equations:" << std::endl
        << "  Iterations: " << std::setw(6) << std::right << iterations
        << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << computing_times[0]
        << std::endl;

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::postprocessing() const
{
  postprocessor->do_postprocessing(solution);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::get_wall_times(std::vector<std::string> & name,
                                          std::vector<double> &      wall_time) const
{
  name.resize(1);
  std::vector<std::string> names = {"Linear system"};
  name                           = names;

  wall_time.resize(1);
  wall_time[0] = computing_times[0];
}

template class DriverSteady<2, float>;
template class DriverSteady<2, double>;

template class DriverSteady<3, float>;
template class DriverSteady<3, double>;

} // namespace Structure
