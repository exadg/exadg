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

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_interface.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/time_integration/driver_steady_problems.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
DriverSteadyProblems<dim, Number>::DriverSteadyProblems(
  std::shared_ptr<Operator>                       operator_,
  Parameters const &                              param_,
  MPI_Comm const &                                mpi_comm_,
  bool const                                      is_test_,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_)
  : pde_operator(operator_),
    param(param_),
    mpi_comm(mpi_comm_),
    is_test(is_test_),
    timer_tree(new TimerTree()),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    postprocessor(postprocessor_),
    iterations({0, {0, 0}})
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
LinearAlgebra::distributed::Vector<Number> const &
DriverSteadyProblems<dim, Number>::get_velocity() const
{
  return solution.block(0);
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
DriverSteadyProblems<dim, Number>::get_timings() const
{
  return timer_tree;
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
  pde_operator->prescribe_initial_conditions(solution.block(0), solution.block(1), time);
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::solve(double const time, bool unsteady_problem)
{
  Timer timer;
  timer.restart();

  postprocessing(time, unsteady_problem);

  do_solve(time, unsteady_problem);

  postprocessing(time, unsteady_problem);

  timer_tree->insert({"DriverSteady"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::do_solve(double const time, bool unsteady_problem)
{
  if(iterations.first == 0)
    global_timer.restart();

  Timer timer;
  timer.restart();

  if(print_solver_info(time, unsteady_problem))
    pcout << std::endl << "Solve steady state problem:" << std::endl;

  // Update divergence and continuity penalty operator in case
  // that these terms are added to the monolithic system of equations
  // instead of applying these terms in a postprocessing step.
  if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
  {
    AssertThrow(this->param.apply_penalty_terms_in_postprocessing_step == false,
                ExcMessage(
                  "Penalty terms have to be applied in momentum equation for steady problems."));

    if(this->param.use_divergence_penalty == true)
      pde_operator->update_divergence_penalty_operator(solution.block(0));
    if(this->param.use_continuity_penalty == true)
      pde_operator->update_continuity_penalty_operator(solution.block(0));
  }

  // linear problem
  if(this->param.linear_problem_has_to_be_solved())
  {
    // calculate rhs vector
    pde_operator->rhs_stokes_problem(rhs_vector, time);

    // solve coupled system of equations
    unsigned int const n_iter = pde_operator->solve_linear_stokes_problem(
      solution, rhs_vector, this->param.update_preconditioner_coupled, time);

    if(print_solver_info(time, unsteady_problem) and not(this->is_test))
      print_solver_info_linear(pcout, n_iter, timer.wall_time());

    iterations.first += 1;
    std::get<1>(iterations.second) += n_iter;
  }
  else // nonlinear problem
  {
    VectorType rhs(solution.block(0));
    rhs = 0.0;
    if(this->param.right_hand_side)
      pde_operator->evaluate_add_body_force_term(rhs, time);

    // Newton solver
    auto const iter = pde_operator->solve_nonlinear_problem(
      solution, rhs, this->param.update_preconditioner_coupled, time);

    if(print_solver_info(time, unsteady_problem) and not(this->is_test))
      print_solver_info_nonlinear(pcout, std::get<0>(iter), std::get<1>(iter), timer.wall_time());

    iterations.first += 1;
    std::get<0>(iterations.second) += std::get<0>(iter);
    std::get<1>(iterations.second) += std::get<1>(iter);
  }

  pde_operator->adjust_pressure_level_if_undefined(solution.block(1), time);

  timer_tree->insert({"DriverSteady", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
bool
DriverSteadyProblems<dim, Number>::print_solver_info(double const time, bool unsteady_problem) const
{
  return !unsteady_problem || param.solver_info_data.write(this->global_timer.wall_time(),
                                                           time - param.start_time,
                                                           iterations.first + 1);
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.linear_problem_has_to_be_solved())
  {
    names = {"Coupled system"};
    iterations_avg.resize(1);
    iterations_avg[0] =
      (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
  }
  else // nonlinear system of equations in momentum step
  {
    names = {"Coupled system (nonlinear)",
             "Coupled system (linear accumulated)",
             "Coupled system (linear per nonlinear)"};

    iterations_avg.resize(3);
    iterations_avg[0] =
      (double)std::get<0>(iterations.second) / std::max(1., (double)iterations.first);
    iterations_avg[1] =
      (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
    if(iterations_avg[0] > std::numeric_limits<double>::min())
      iterations_avg[2] = iterations_avg[1] / iterations_avg[0];
    else
      iterations_avg[2] = iterations_avg[1];
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::postprocessing(double const time, bool unsteady_problem) const
{
  Timer timer;
  timer.restart();

  if(unsteady_problem)
    postprocessor->do_postprocessing(solution.block(0),
                                     solution.block(1),
                                     (Number)time,
                                     (int)iterations.first);
  else
    postprocessor->do_postprocessing(solution.block(0), solution.block(1));

  timer_tree->insert({"DriverSteady", "Postprocessing"}, timer.wall_time());
}


template class DriverSteadyProblems<2, float>;
template class DriverSteadyProblems<2, double>;

template class DriverSteadyProblems<3, float>;
template class DriverSteadyProblems<3, double>;

} // namespace IncNS
} // namespace ExaDG
