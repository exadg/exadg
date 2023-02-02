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
      linear_solver(linear_solver_in)
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

      // update linear operator (set linearization point)
      linear_operator.set_solution_linearization(solution);

      // determine whether to update the operator/preconditioner of the linearized problem
      bool const update_now =
        update.do_update and (newton_iterations % update.update_every_newton_iter == 0);

      // update the preconditioner
      linear_solver.update_preconditioner(update_now);

      // solve linear problem
      unsigned int const n_iter_linear = linear_solver.solve(increment, residual);

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
                  dealii::ExcMessage("Damped Newton iteration did not converge. "
                                     "Maximum number of iterations exceeded!"));

      // update solution and residual
      solution = temporary;
      norm_r   = norm_r_damp;

      // increment iteration counter
      ++newton_iterations;
      linear_iterations += n_iter_linear;
    }

    AssertThrow(norm_r <= this->solver_data.abs_tol || norm_r / norm_r_0 <= solver_data.rel_tol,
                dealii::ExcMessage(
                  "Newton solver failed to solve nonlinear problem to given tolerance. "
                  "Maximum number of iterations exceeded!"));

    if(update.do_update and update.update_once_converged)
    {
      // update linear operator (set linearization point)
      linear_operator.set_solution_linearization(solution);
      // update preconditioner
      linear_solver.update_preconditioner(true);
    }

    return std::tuple<unsigned int, unsigned int>(newton_iterations, linear_iterations);
  }

private:
  SolverData          solver_data;
  NonlinearOperator & nonlinear_operator;
  LinearOperator &    linear_operator;
  LinearSolver &      linear_solver;
};

} // namespace Newton
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_ */
