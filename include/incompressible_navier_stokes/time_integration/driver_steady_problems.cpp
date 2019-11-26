/*
 * driver_steady_problems.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include "driver_steady_problems.h"

#include "../spatial_discretization/interface.h"
#include "../user_interface/input_parameters.h"
#include "functionalities/set_zero_mean_value.h"

namespace IncNS
{
template<typename Number>
DriverSteadyProblems<Number>::DriverSteadyProblems(std::shared_ptr<OperatorBase> operator_base_in,
                                                   std::shared_ptr<OperatorPDE>  operator_in,
                                                   InputParameters const &       param_in)
  : operator_base(operator_base_in),
    pde_operator(operator_in),
    param(param_in),
    computing_times(1),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
}

template<typename Number>
void
DriverSteadyProblems<Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by using the analytical solution
  // or a guess of the velocity and pressure field
  initialize_solution();
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
DriverSteadyProblems<Number>::get_velocity() const
{
  return solution.block(0);
}

template<typename Number>
void
DriverSteadyProblems<Number>::initialize_vectors()
{
  // solution
  pde_operator->initialize_block_vector_velocity_pressure(solution);

  // rhs_vector
  if(this->param.equation_type == EquationType::Stokes)
    pde_operator->initialize_block_vector_velocity_pressure(rhs_vector);
}

template<typename Number>
void
DriverSteadyProblems<Number>::initialize_solution()
{
  double time = 0.0;
  operator_base->prescribe_initial_conditions(solution.block(0), solution.block(1), time);
}

template<typename Number>
void
DriverSteadyProblems<Number>::solve()
{
  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  Timer timer;
  timer.restart();

  // Update divergence and continuity penalty operator in case
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

  unsigned int N_iter_nonlinear = 0;
  unsigned int N_iter_linear    = 0;

  // linear problem
  if(this->param.linear_problem_has_to_be_solved())
  {
    // calculate rhs vector
    pde_operator->rhs_stokes_problem(rhs_vector);

    // solve coupled system of equations
    N_iter_linear =
      pde_operator->solve_linear_stokes_problem(solution,
                                                rhs_vector,
                                                this->param.update_preconditioner_coupled);
  }
  else // nonlinear problem
  {
    VectorType rhs(solution.block(0));
    rhs = 0.0;
    if(this->param.right_hand_side)
      operator_base->evaluate_add_body_force_term(rhs, 0.0 /* time */);

    // Newton solver
    pde_operator->solve_nonlinear_steady_problem(
      solution, rhs, this->param.update_preconditioner_coupled, N_iter_nonlinear, N_iter_linear);
  }

  // special case: pure Dirichlet BC's
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      this->operator_base->shift_pressure(solution.block(1));
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      set_zero_mean_value(solution.block(1));
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      this->operator_base->shift_pressure_mean_value(solution.block(1));
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }

  computing_times[0] += timer.wall_time();

  // write output
  if(this->param.equation_type == EquationType::Stokes)
  {
    pcout << std::endl
          << "Solve linear problem:" << std::endl
          << "  Iterations:   " << std::setw(12) << std::right << N_iter_linear << std::endl
          << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(4)
          << computing_times[0] << std::endl;
  }
  else // nonlinear problem
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

  pcout << std::endl << "... done!" << std::endl;
}

template<typename Number>
void
DriverSteadyProblems<Number>::get_wall_times(std::vector<std::string> & name,
                                             std::vector<double> &      wall_time) const
{
  name.resize(1);
  std::vector<std::string> names = {"Coupled system"};
  name                           = names;

  wall_time.resize(1);
  for(unsigned int i = 0; i < this->computing_times.size(); ++i)
  {
    wall_time[i] = this->computing_times[i];
  }
}

template<typename Number>
void
DriverSteadyProblems<Number>::solve_steady_problem()
{
  postprocessing();

  solve();

  postprocessing();
}

template<typename Number>
void
DriverSteadyProblems<Number>::postprocessing()
{
  this->operator_base->do_postprocessing_steady_problem(solution.block(0), solution.block(1));
}

// instantiations

template class DriverSteadyProblems<float>;
template class DriverSteadyProblems<double>;

} // namespace IncNS
