/*
 * driver_steady_problems.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include "driver_steady_problems.h"

#include "../interface_space_time/operator.h"
#include "../user_interface/input_parameters.h"
#include "functionalities/set_zero_mean_value.h"

namespace IncNS
{
template<int dim, typename Number>
DriverSteadyProblems<dim, Number>::DriverSteadyProblems(
  std::shared_ptr<OperatorBase> operator_base_in,
  std::shared_ptr<OperatorPDE>  operator_in,
  InputParameters<dim> const &  param_in)
  : operator_base(operator_base_in), pde_operator(operator_in), param(param_in), total_time(0.0)
{
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by using the analytical solution
  // or a guess of the velocity and pressure field
  initialize_solution();
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::initialize_vectors()
{
  // solution
  pde_operator->initialize_block_vector_velocity_pressure(solution);

  // rhs_vector
  if(this->param.equation_type == EquationType::Stokes)
    pde_operator->initialize_block_vector_velocity_pressure(rhs_vector);
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::initialize_solution()
{
  double time = 0.0;
  operator_base->prescribe_initial_conditions(solution.block(0), solution.block(1), time);
}


template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::solve()
{
  Timer timer;
  timer.restart();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Solving steady state problem ..." << std::endl;

  // Update divegence and continuity penalty operator in case
  // that these terms are added to the monolithic system of equations
  // instead of applying these terms in a postprocessing step.
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      if(this->param.use_divergence_penalty == true)
      {
        pde_operator->update_divergence_penalty_operator(solution.block(0));
      }
      if(this->param.use_continuity_penalty == true)
      {
        pde_operator->update_continuity_penalty_operator(solution.block(0));
      }
    }
  }

  // Steady Stokes equations
  if(this->param.equation_type == EquationType::Stokes)
  {
    // calculate rhs vector
    pde_operator->rhs_stokes_problem(rhs_vector);

    // solve coupled system of equations
    unsigned int iterations = pde_operator->solve_linear_stokes_problem(solution, rhs_vector);
    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std::endl
                << "Solve linear Stokes problem:" << std::endl
                << "  Iterations:   " << std::setw(12) << std::right << iterations << std::endl
                << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(4)
                << timer.wall_time() << std::endl;
    }
  }
  else // Steady Navier-Stokes equations
  {
    // Newton solver
    unsigned int newton_iterations;
    unsigned int linear_iterations;
    pde_operator->solve_nonlinear_steady_problem(solution, newton_iterations, linear_iterations);

    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std::endl
                << "Solve nonlinear Navier-Stokes problem:" << std::endl
                << "  Newton iterations:      " << std::setw(12) << std::right << newton_iterations
                << std::endl
                << "  Linear iterations (avg):" << std::setw(12) << std::scientific
                << std::setprecision(4) << std::right
                << double(linear_iterations) / double(newton_iterations) << std::endl
                << "  Linear iterations (tot):" << std::setw(12) << std::scientific
                << std::setprecision(4) << std::right << linear_iterations << std::endl
                << "  Wall time [s]:          " << std::setw(12) << std::scientific
                << std::setprecision(4) << timer.wall_time() << std::endl;
    }
  }

  // special case: pure Dirichlet BC's
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.error_data.analytical_solution_available == true)
      operator_base->shift_pressure(solution.block(1));
    else // analytical_solution_available == false
      set_zero_mean_value(solution.block(1));
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::solve_steady_problem()
{
  global_timer.restart();

  postprocessing();

  solve();

  postprocessing();

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::postprocessing()
{
  pde_operator->do_postprocessing_steady_problem(solution.block(0), solution.block(1));
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::analyze_computing_times() const
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

// instantiations
#include <navierstokes/config.h>

// float
#if DIM_2 && OP_FLOAT
template class DriverSteadyProblems<2, float>;
#endif
#if DIM_3 && OP_FLOAT
template class DriverSteadyProblems<3, float>;
#endif

// double
#if DIM_2 && OP_DOUBLE
template class DriverSteadyProblems<2, double>;
#endif
#if DIM_3 && OP_DOUBLE
template class DriverSteadyProblems<3, double>;
#endif

} // namespace IncNS
