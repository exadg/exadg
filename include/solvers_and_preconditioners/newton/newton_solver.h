/*
 * NewtonSolver.h
 *
 *  Created on: Jun 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_

#include "newton_solver_data.h"

template<typename Vector,
         typename NonlinearOperator,
         typename LinearOperator,
         typename SolverLinearizedProblem>
class NewtonSolver
{
public:
  NewtonSolver(NewtonSolverData const &  solver_data_in,
               NonlinearOperator &       nonlinear_operator_in,
               LinearOperator &          linear_operator_in,
               SolverLinearizedProblem & linear_solver_in)
    : solver_data(solver_data_in),
      nonlinear_operator(nonlinear_operator_in),
      linear_operator(linear_operator_in),
      linear_solver(linear_solver_in)
  {
    nonlinear_operator.initialize_vector_for_newton_solver(residual);
    nonlinear_operator.initialize_vector_for_newton_solver(increment);
    nonlinear_operator.initialize_vector_for_newton_solver(tmp);
  }

  void
  solve(Vector & dst, unsigned int & newton_iterations, unsigned int & linear_iterations)
  {
    // evaluate residual using the given estimate of the solution
    nonlinear_operator.evaluate_nonlinear_residual(residual, dst);

    double norm_r   = residual.l2_norm();
    double norm_r_0 = norm_r;

    // reset average_linear_iterations
    linear_iterations = 0.0;

    // Newton iteration
    unsigned int n_iter = 0;

    while(norm_r > this->solver_data.abs_tol && norm_r / norm_r_0 > solver_data.rel_tol &&
          n_iter < solver_data.max_iter)
    {
      // reset increment
      increment = 0.0;

      // multiply by -1.0 since the linearized problem is "LinearMatrix * increment = - residual"
      residual *= -1.0;

      // solve linear problem
      dst.update_ghost_values();
      linear_operator.set_solution_linearization(dst);
      linear_iterations += linear_solver.solve(increment, residual);

      if(false) //(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "  Number of linear solver iterations: " << linear_iterations << std::endl;

      // damped Newton scheme
      double       omega      = 1.0; // damping factor
      double       tau        = 0.5; // another parameter (has to be smaller than 1)
      double       norm_r_tmp = 1.0; // norm of residual using temporary solution
      unsigned int n_iter_tmp = 0, N_ITER_TMP_MAX = 10; // iteration counts for damping scheme

      do
      {
        // calculate temporary solution
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        tmp.equ(1.0, dst, omega, increment);
#pragma GCC diagnostic pop

        // evaluate residual using the temporary solution
        nonlinear_operator.evaluate_nonlinear_residual(residual, tmp);

        // calculate norm of residual (for temporary solution)
        norm_r_tmp = residual.l2_norm();

        // reduce step length
        omega = omega / 2.0;

        // increment counter
        n_iter_tmp++;
      } while(norm_r_tmp >= (1.0 - tau * omega) * norm_r && n_iter_tmp < N_ITER_TMP_MAX);

      AssertThrow(norm_r_tmp < (1.0 - tau * omega) * norm_r,
                  ExcMessage("Damped Newton iteration did not converge. "
                             "Maximum number of iterations exceeded!"));

      //      if(norm_r_tmp >= (1.0-tau*omega)*norm_r)
      //      {
      //        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      //          std::cout << "Damped Newton iteration did not converge." << std::endl
      //                    << "Maximum number of iterations exceeded!" << std::endl;
      //      }

      // update solution and residual
      dst    = tmp;
      norm_r = norm_r_tmp;

      // increment iteration counter
      ++n_iter;
    }

    AssertThrow(norm_r <= this->solver_data.abs_tol || norm_r / norm_r_0 <= solver_data.rel_tol,
                ExcMessage("Newton solver failed to solve nonlinear problem to given tolerance. "
                           "Maximum number of iterations exceeded!"));

    //    if(norm_r > this->solver_data.abs_tol && norm_r/norm_r_0 > solver_data.rel_tol)
    //    {
    //      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    //        std::cout << "Newton solver failed to solve nonlinear problem to given tolerance." <<
    //        std::endl
    //                  << "Maximum number of iterations exceeded!" << std::endl;
    //    }

    newton_iterations = n_iter;

    return;
  }

private:
  NewtonSolverData          solver_data;
  NonlinearOperator &       nonlinear_operator;
  LinearOperator &          linear_operator;
  SolverLinearizedProblem & linear_solver;
  Vector                    residual, increment, tmp;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_ */
