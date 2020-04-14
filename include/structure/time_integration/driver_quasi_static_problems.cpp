/*
 * driver_quasi_static_problems.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "driver_quasi_static_problems.h"

namespace Structure
{
template<int dim, typename Number>
DriverQuasiStatic<dim, Number>::DriverQuasiStatic(
  std::shared_ptr<Operator<dim, Number>>      operator_in,
  std::shared_ptr<PostProcessor<dim, Number>> postprocessor_in,
  InputParameters const &                     param_in,
  MPI_Comm const &                            mpi_comm_in)
  : pde_operator(operator_in),
    postprocessor(postprocessor_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    computing_times(1),
    time(param.end_time, param.delta_t, mpi_comm_in)
{
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::solve_problem()
{
  pcout << std::endl << "Solving quasi-static problem ..." << std::endl;

  // perform post processing for initial condition
  postprocessing();

  // perform time loop
  for(; !time.finished(); time.do_increment())
  {
    // print information to the current time step
    time.print_header();

    // compute deformation increment
    unsigned int const iterations = solve_primary();

    if(param.time_step_control)
      adapt_time_step_size(iterations, time.get_delta_t(), 300, 1);

    // move mesh as well as update matrixfree, solvers, and preconditioners
    if(param.updated_formulation)
      pde_operator->move_mesh(solution);

    // perform postprocessing
    postprocessing();
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::adapt_time_step_size(unsigned int n_current_iteration,
                                                     double       current_delta_t,
                                                     unsigned int ideal_iterations,
                                                     double       p)
{
  if(current_delta_t > 1.0e-3)
  {
    double A = double(ideal_iterations);
    double B = double(n_current_iteration);

    std::cout << "n_current_iteration = " << n_current_iteration << std::endl;
    std::cout << "ideal_iterations = " << ideal_iterations << std::endl;
    std::cout << "current_delta_t = " << current_delta_t << std::endl;

    auto R = A / B;

    std::cout << "R = " << R << std::endl;

    double alpha       = std::pow(R, p);
    double new_delta_t = alpha * current_delta_t;

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "new_delta_t = " << new_delta_t << std::endl;

    time.set_delta_t(new_delta_t);
  }
  else
  {
    double new_delta_t = current_delta_t;
    time.set_delta_t(new_delta_t);
  }
}


template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::initialize_vectors()
{
  // solution
  pde_operator->initialize_dof_vector(solution);

  // rhs_vector
  pde_operator->initialize_dof_vector(rhs_vector);
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::initialize_solution()
{
  double time = 0.0;
  pde_operator->prescribe_initial_conditions(solution, time);
}

template<int dim, typename Number>
unsigned int
DriverQuasiStatic<dim, Number>::solve_primary()
{
  Timer timer;
  timer.restart();

  // calculate rhs vector
  pde_operator->rhs(rhs_vector, time.get_current_time());

  unsigned int N_iter_nonlinear = 0;
  unsigned int N_iter_linear    = 0;

  // solve system of equations
  if(param.large_deformation)
  {
    pde_operator->solve_nonlinear(solution,
                                  rhs_vector,
                                  time.get_current_time(),
                                  /* update_preconditioner = */ true,
                                  N_iter_nonlinear,
                                  N_iter_linear);
  }
  else
  {
    N_iter_linear = pde_operator->solve_linear(solution,
                                               rhs_vector,
                                               time.get_current_time(),
                                               /* update_preconditioner = */ true);
  }

  computing_times[0] += timer.wall_time();

  // solver info output
  if(param.large_deformation)
  {
    double N_iter_linear_avg =
      (N_iter_nonlinear > 0) ? double(N_iter_linear) / double(N_iter_nonlinear) : N_iter_linear;

    pcout << std::endl
          << "Solve nonlinear problem:" << std::endl
          << "  Newton iterations:      " << std::setw(12) << std::right << N_iter_nonlinear
          << std::endl
          << "  Linear iterations (avg):" << std::setw(12) << std::scientific
          << std::setprecision(4) << std::right << N_iter_linear_avg << std::endl
          << "  Linear iterations (tot):" << std::setw(12) << std::scientific
          << std::setprecision(4) << std::right << N_iter_linear << std::endl
          << "  Wall time [s]:          " << std::setw(12) << std::scientific
          << std::setprecision(4) << computing_times[0] << std::endl;
  }
  else
  {
    pcout << std::endl
          << "Solve linear problem:" << std::endl
          << "  Iterations:   " << std::setw(12) << std::right << N_iter_linear << std::endl
          << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(4)
          << computing_times[0] << std::endl;
  }

  return N_iter_linear;
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::postprocessing() const
{
  postprocessor->do_postprocessing(solution);
  norm_vector.push_back(this->solution.norm_sqr());
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::get_wall_times(std::vector<std::string> & name,
                                               std::vector<double> &      wall_time) const
{
  name.resize(1);
  std::vector<std::string> names = {"Linear system"};
  name                           = names;

  wall_time.resize(1);
  wall_time[0] = computing_times[0];

  // TODO: move it to some more useful place
  for(auto i : norm_vector)
    pcout << "Norm_vector_component: " << i << std::endl;
}

template class DriverQuasiStatic<2, float>;
template class DriverQuasiStatic<2, double>;

template class DriverQuasiStatic<3, float>;
template class DriverQuasiStatic<3, double>;

} // namespace Structure
