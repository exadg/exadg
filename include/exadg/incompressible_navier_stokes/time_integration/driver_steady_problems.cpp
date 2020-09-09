/*
 * driver_steady_problems.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_interface.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/driver_steady_problems.h>
#include <exadg/incompressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/utilities/print_throughput.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
DriverSteadyProblems<dim, Number>::DriverSteadyProblems(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  MPI_Comm const &                                mpi_comm_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : pde_operator(operator_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    timer_tree(new TimerTree()),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    postprocessor(postprocessor_in)
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
DriverSteadyProblems<dim, Number>::solve()
{
  Timer timer;
  timer.restart();

  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  // Update divergence and continuity penalty operator in case
  // that these terms are added to the monolithic system of equations
  // instead of applying these terms in a postprocessing step.
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      pde_operator->update_divergence_penalty_operator(solution.block(0));
    if(this->param.use_continuity_penalty == true)
      pde_operator->update_continuity_penalty_operator(solution.block(0));
  }

  // linear problem
  if(this->param.linear_problem_has_to_be_solved())
  {
    // calculate rhs vector
    pde_operator->rhs_stokes_problem(rhs_vector);

    // solve coupled system of equations
    unsigned int const N_iter_linear =
      pde_operator->solve_linear_stokes_problem(solution,
                                                rhs_vector,
                                                this->param.update_preconditioner_coupled);

    print_solver_info_linear(pcout, N_iter_linear, timer.wall_time());
  }
  else // nonlinear problem
  {
    VectorType rhs(solution.block(0));
    rhs = 0.0;
    if(this->param.right_hand_side)
      pde_operator->evaluate_add_body_force_term(rhs, 0.0 /* time */);

    // Newton solver
    auto const iter =
      pde_operator->solve_nonlinear_steady_problem(solution,
                                                   rhs,
                                                   this->param.update_preconditioner_coupled);

    print_solver_info_nonlinear(pcout, std::get<0>(iter), std::get<1>(iter), timer.wall_time());
  }

  pde_operator->adjust_pressure_level_if_undefined(solution.block(1), 0.0 /* time */);

  pcout << std::endl << "... done!" << std::endl;

  timer_tree->insert({"DriverSteady", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::solve_steady_problem()
{
  Timer timer;
  timer.restart();

  postprocessing();

  solve();

  postprocessing();

  timer_tree->insert({"DriverSteady"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::postprocessing() const
{
  Timer timer;
  timer.restart();

  postprocessor->do_postprocessing(solution.block(0), solution.block(1));

  timer_tree->insert({"DriverSteady", "Postprocessing"}, timer.wall_time());
}


template class DriverSteadyProblems<2, float>;
template class DriverSteadyProblems<2, double>;

template class DriverSteadyProblems<3, float>;
template class DriverSteadyProblems<3, double>;

} // namespace IncNS
} // namespace ExaDG
