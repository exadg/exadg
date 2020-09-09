/*
 * newton_solver.h
 *
 *  Created on: Jun 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/newton/newton_solver_data.h>

namespace ExaDG
{
namespace Newton
{
using namespace dealii;

template<typename VectorType,
         typename NonlinearOperator,
         typename LinearOperator,
         typename LinearSolver>
class Solver
{
public:
  Solver(SolverData const &  solver_data_in,
         NonlinearOperator & nonlinear_operator_in,
         LinearOperator &    linear_operator_in,
         LinearSolver &      linear_solver_in)
    : solver_data(solver_data_in),
      nonlinear_operator(nonlinear_operator_in),
      linear_operator(linear_operator_in),
      linear_solver(linear_solver_in),
      linear_iterations_last(0)
  {
  }

  std::tuple<unsigned int /* Newton iter */, unsigned int /* accumulated linear iter */>
  solve(VectorType & solution, UpdateData const & update)
  {
    unsigned int newton_iterations = 0, linear_iterations = 0;

    VectorType residual, increment, temporary;
    residual.reinit(solution);
    increment.reinit(solution);
    temporary.reinit(solution);

    // evaluate residual using initial guess of solution
    nonlinear_operator.evaluate_residual(residual, solution);

    double norm_r   = residual.l2_norm();
    double norm_r_0 = norm_r;

    while(norm_r > this->solver_data.abs_tol && norm_r / norm_r_0 > solver_data.rel_tol &&
          newton_iterations < solver_data.max_iter)
    {
      // reset increment
      increment = 0.0;

      // multiply by -1.0 since the linearized problem is "linear_operator * increment = - residual"
      residual *= -1.0;

      // set linearization point
      linear_operator.set_solution_linearization(solution);

      // determine whether to update the operator/preconditioner of the linearized problem
      bool const threshold_exceeded = (newton_iterations % update.threshold_newton_iter == 0) ||
                                      (linear_iterations_last > update.threshold_linear_iter);

      // solve linear problem
      linear_iterations_last =
        linear_solver.solve(increment, residual, update.do_update && threshold_exceeded);

      // damped Newton scheme
      double             omega         = 1.0; // damping factor (begin with 1)
      double             norm_r_damp   = 1.0; // norm of residual using temporary solution
      unsigned int       n_iter_damp   = 0;   // counts iteration of damping scheme
      unsigned int const max_iter_damp = 10;  // max iterations of damping scheme
      double const       tau           = 0.5; // a parameter (has to be smaller than 1)
      do
      {
        // add increment to solution vector but scale by a factor omega <= 1
        temporary = solution;
        temporary.add(omega, increment);

        // evaluate residual using the temporary solution
        nonlinear_operator.evaluate_residual(residual, temporary);

        // calculate norm of residual (for temporary solution)
        norm_r_damp = residual.l2_norm();

        // reduce step length
        omega = omega / 2.0;

        // increment counter
        n_iter_damp++;
      } while(norm_r_damp >= (1.0 - tau * omega) * norm_r && n_iter_damp < max_iter_damp);

      AssertThrow(norm_r_damp < (1.0 - tau * omega) * norm_r,
                  ExcMessage("Damped Newton iteration did not converge. "
                             "Maximum number of iterations exceeded!"));

      // update solution and residual
      solution = temporary;
      norm_r   = norm_r_damp;

      // increment iteration counter
      ++newton_iterations;
      linear_iterations += linear_iterations_last;
    }

    AssertThrow(norm_r <= this->solver_data.abs_tol || norm_r / norm_r_0 <= solver_data.rel_tol,
                ExcMessage("Newton solver failed to solve nonlinear problem to given tolerance. "
                           "Maximum number of iterations exceeded!"));

    return std::tuple<unsigned int, unsigned int>(newton_iterations, linear_iterations);
  }

private:
  SolverData          solver_data;
  NonlinearOperator & nonlinear_operator;
  LinearOperator &    linear_operator;
  LinearSolver &      linear_solver;

  unsigned int linear_iterations_last;
};

} // namespace Newton
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_ */
