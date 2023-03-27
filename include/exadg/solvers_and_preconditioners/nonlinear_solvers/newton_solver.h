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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NONLINEAR_SOLVER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NONLINEAR_SOLVER_H_

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/nonlinear_solvers/newton_solver_data.h>

namespace ExaDG
{
namespace NonlinearSolver
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

  std::tuple<unsigned int /* nonlinear iter */, unsigned int /* accumulated linear iter */>
  solve(VectorType & solution, UpdateData const & update)
  {
    unsigned int nonlinear_iterations = 0, linear_iterations = 0;

    VectorType residual_rhs, nonlinear_solver_output, temporary_solution;
    residual_rhs.reinit(solution);
    nonlinear_solver_output.reinit(solution);
    temporary_solution.reinit(solution);

    // evaluate residual using initial guess of solution
    nonlinear_operator.evaluate_residual(residual_rhs, solution);

    double norm_r   = residual_rhs.l2_norm();
    double norm_r_0 = norm_r;

    while(norm_r > this->solver_data.abs_tol && norm_r / norm_r_0 > solver_data.rel_tol &&
          nonlinear_iterations < solver_data.max_iter)
    {
      // reset iterate and rhs of nonlinear solver
      if(solver_data.nonlinear_solver_type == SolverType::Newton)
      {
        // reset increment
        nonlinear_solver_output = 0.0;

        // overwrite residual with right-hand side for Newton solver
        // multiply by -1.0 since the linearized problem is "linear_operator * increment = -
        // residual"
        residual_rhs *= -1.0;
      }
      else if(solver_data.nonlinear_solver_type == SolverType::Picard)
      {
        // set initial guess
        nonlinear_solver_output = solution;

        // overwrite residual with right-hand side for Picard solver
        nonlinear_operator.evaluate_rhs_picard(residual_rhs);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Nonlinear solver not defined."));
      }

      // update linear operator (set linearization point)
      linear_operator.set_solution_linearization(solution);

      // determine whether to update the operator/preconditioner of the linearized problem
      bool const update_now =
        update.do_update and (nonlinear_iterations % update.update_every_nonlinear_iter == 0);

      // update the preconditioner
      linear_solver.update_preconditioner(update_now);

      // solve linear problem
      unsigned int const n_iter_linear = linear_solver.solve(nonlinear_solver_output, residual_rhs);

      // damped nonlinear scheme
      double             omega         = 1.0; // damping factor (begin with 1)
      double             norm_r_damp   = 1.0; // norm of residual using temporary solution
      unsigned int       n_iter_damp   = 0;   // counts iteration of damping scheme
      unsigned int const max_iter_damp = 10;  // max iterations of damping scheme
      double const       tau           = 0.5; // a parameter (has to be smaller than 1)
      do
      {
        // add increment to solution vector but scale by a factor omega <= 1
        temporary_solution = solution;
        if(solver_data.nonlinear_solver_type == SolverType::Newton)
        {
          // x^{*} = x^{k} + omega * (x^{k+1} - x^{k})
          temporary_solution.add(omega, nonlinear_solver_output);
        }
        else if(solver_data.nonlinear_solver_type == SolverType::Picard)
        {
          // x^{*} = (1 - omega) * x^{k} + omega * x^{k+1}
          temporary_solution.sadd((1.0 - omega), omega, nonlinear_solver_output);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Nonlinear solver not defined."));
        }

        // evaluate residual using the temporary solution
        nonlinear_operator.evaluate_residual(residual_rhs, temporary_solution);

        // calculate norm of residual (for temporary solution)
        norm_r_damp = residual_rhs.l2_norm();

        // reduce step length
        omega = omega / 2.0;

        // increment counter
        n_iter_damp++;
      } while(norm_r_damp >= (1.0 - tau * omega) * norm_r && n_iter_damp < max_iter_damp);

      AssertThrow(norm_r_damp < (1.0 - tau * omega) * norm_r,
                  dealii::ExcMessage("Damped nonlinear iteration did not converge. "
                                     "Maximum number of iterations (" +
                                     std::to_string(max_iter_damp) + ") exceeded!"));

      // update solution and residual
      solution = temporary_solution;
      norm_r   = norm_r_damp;

      // increment iteration counter
      ++nonlinear_iterations;
      linear_iterations += n_iter_linear;
    }

    AssertThrow(norm_r <= this->solver_data.abs_tol || norm_r / norm_r_0 <= solver_data.rel_tol,
                dealii::ExcMessage(
                  "Nonlinear solver failed to solve nonlinear problem to given tolerance. "
                  "Maximum number of iterations exceeded!"));

    if(update.do_update and update.update_once_converged)
    {
      // update linear operator (set linearization point)
      linear_operator.set_solution_linearization(solution);
      // update preconditioner
      linear_solver.update_preconditioner(true);
    }

    return std::tuple<unsigned int, unsigned int>(nonlinear_iterations, linear_iterations);
  }

private:
  SolverData          solver_data;
  NonlinearOperator & nonlinear_operator;
  LinearOperator &    linear_operator;
  LinearSolver &      linear_solver;
};

} // namespace NonlinearSolver
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NONLINEAR_SOLVER_H_ */
