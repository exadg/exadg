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
    step_number(1)
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
  pcout << std::endl << "Solving quasi-static problem ..." << std::endl << std::flush;

  // perform post processing for initial solution
  postprocessing();

  // perform time loop
  double       load_factor    = 0.0;
  double       load_increment = param.load_increment;
  double const eps            = 1.e-10;
  while(load_factor < 1.0 - eps)
  {
    unsigned int iterations = 0;

    // compute displacement for new load factor
    if(param.adjust_load_increment)
    {
      // reduce load increment in factors of 2 until the current
      // step can be solved successfully
      bool         success        = false;
      unsigned int re_try_counter = 0;
      while(!success && re_try_counter < 10)
      {
        try
        {
          iterations = solve(load_factor + load_increment);
          success    = true;
          ++re_try_counter;
        }
        catch(...)
        {
          load_increment *= 0.5;
          pcout << std::endl
                << "Could not solve non-linear problem. Reduce load factor to "
                << load_factor + load_increment << std::flush;
        }
      }
    }
    else
    {
      iterations = solve(load_factor + load_increment);
    }

    // increment load factor
    load_factor += load_increment;
    ++step_number;

    // adjust increment for next load step
    if(param.adjust_load_increment)
    {
      if(iterations > 0)
        load_increment *=
          std::pow((double)param.desired_newton_iterations / (double)iterations, 0.5);
    }

    // make sure to hit maximum load exactly
    if(load_factor + load_increment >= 1.0)
      load_increment = 1.0 - load_factor;
  }

  // perform post processing for final solution
  postprocessing();

  pcout << std::endl << "... done!" << std::endl;
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
  pde_operator->prescribe_initial_displacement(solution, 0.0 /* time */);
}

template<int dim, typename Number>
unsigned int
DriverQuasiStatic<dim, Number>::solve(double const load_factor)
{
  Timer timer;
  timer.restart();

  unsigned int N_iter_nonlinear = 0;
  unsigned int N_iter_linear    = 0;

  // solve system of equations
  if(param.large_deformation)
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Solve non-linear problem for load factor = " << std::scientific
          << std::setprecision(4) << load_factor << std::endl
          << "______________________________________________________________________" << std::endl;

    VectorType const_vector;

    bool const update_preconditioner =
      this->param.update_preconditioner &&
      ((this->step_number - 1) % this->param.update_preconditioner_every_time_steps == 0);

    auto const iter = pde_operator->solve_nonlinear(solution,
                                                    const_vector,
                                                    0.0 /*no mass term*/,
                                                    load_factor /* = time */,
                                                    update_preconditioner);

    N_iter_nonlinear = std::get<0>(iter);
    N_iter_linear    = std::get<1>(iter);

    // solver info output
    double N_iter_linear_avg =
      (N_iter_nonlinear > 0) ? double(N_iter_linear) / double(N_iter_nonlinear) : N_iter_linear;

    // clang-format off
    pcout << std::endl
          << "  Newton iterations:      " << std::setw(12) << std::right << N_iter_nonlinear << std::endl
          << "  Linear iterations (avg):" << std::setw(12) << std::scientific << std::setprecision(4) << std::right << N_iter_linear_avg << std::endl
          << "  Linear iterations (tot):" << std::setw(12) << std::scientific << std::setprecision(4) << std::right << N_iter_linear << std::endl
          << "  Wall time [s]:          " << std::setw(12) << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
    // clang-format on
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  computing_times[0] += timer.wall_time();

  return N_iter_nonlinear;
}

template<int dim, typename Number>
void
DriverQuasiStatic<dim, Number>::postprocessing() const
{
  postprocessor->do_postprocessing(solution);
}

template class DriverQuasiStatic<2, float>;
template class DriverQuasiStatic<2, double>;

template class DriverQuasiStatic<3, float>;
template class DriverQuasiStatic<3, double>;

} // namespace Structure
