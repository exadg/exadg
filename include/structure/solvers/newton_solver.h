#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_

// TODO this code seems to be copied from include/solvers_and_preconditioners/newton/

// deal.II
#include <deal.II/numerics/data_out.h>

#include "newton_solver_data.h"

template<typename VectorType, typename NonlinearOperator, typename SolverLinearizedProblem, int dim>
class NewtonSolver
{
public:
  NewtonSolver(NewtonSolverData const &  solver_data_in,
               NonlinearOperator &       nonlinear_operator_in,
               SolverLinearizedProblem & linear_solver_in,
               DoFHandler<dim> const &   dof_handler)
    : solver_data(solver_data_in),
      nonlinear_operator(nonlinear_operator_in),
      linear_solver(linear_solver_in),
      dof_handler(dof_handler),
      counter(0)
  {
    nonlinear_operator.initialize_dof_vector(residual);
    nonlinear_operator.initialize_dof_vector(increment);
    nonlinear_operator.initialize_dof_vector(tmp);
  }

  void
  solve(VectorType &       dst,
        unsigned int &     newton_iterations,
        unsigned int &     linear_iterations,
        bool const         update_preconditioner_linear_solver,
        unsigned int const update_preconditioner_every_newton_iter)
  {
    VectorType rhs;
    this->solve(dst,
                rhs,
                newton_iterations,
                linear_iterations,
                update_preconditioner_linear_solver,
                update_preconditioner_every_newton_iter);
  }

  void
  solve(VectorType &       dst,
        VectorType const & rhs,
        unsigned int &     newton_iterations,
        unsigned int &     linear_iterations,
        bool const         update_preconditioner_linear_solver,
        unsigned int const update_preconditioner_every_newton_iter)
  {
    const bool constant_rhs = rhs.size() > 0;

    if(this->solver_data.write_debug_vtk)
      this->debug(dst);

    // evaluate residual using the given estimate of the solution
    nonlinear_operator.evaluate_nonlinear_residual(residual, dst);

    if(constant_rhs)
      residual -= rhs;

    double norm_r   = residual.l2_norm();
    double norm_r_0 = norm_r;

    // Accumulated linear iterations
    newton_iterations = 0;
    linear_iterations = 0;

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
      nonlinear_operator.set_solution_linearization(dst);
      bool const do_update = update_preconditioner_linear_solver &&
                             (n_iter % update_preconditioner_every_newton_iter == 0);
      linear_iterations += linear_solver.solve(increment, residual, do_update);

      // damped Newton scheme
      double       omega      = 1.0; // damping factor
      double       tau        = 0.1; // another parameter (has to be smaller than 1)
      double       norm_r_tmp = 1.0; // norm of residual using temporary solution
      unsigned int n_iter_tmp = 0, N_ITER_TMP_MAX = 100; // iteration counts for damping scheme

      do
      {
        // calculate temporary solution
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        tmp.equ(1.0, dst, omega, increment);
#pragma GCC diagnostic pop

        // evaluate residual using the temporary solution
        nonlinear_operator.evaluate_nonlinear_residual(residual, tmp);
        if(constant_rhs)
          residual -= rhs;

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

      // update solution and residual
      dst    = tmp;
      norm_r = norm_r_tmp;

      if(this->solver_data.write_debug_vtk)
        this->debug(dst);

      // increment iteration counter
      ++n_iter;
    }

    AssertThrow(norm_r <= this->solver_data.abs_tol || norm_r / norm_r_0 <= solver_data.rel_tol,
                ExcMessage("Newton solver failed to solve nonlinear problem to given tolerance. "
                           "Maximum number of iterations exceeded!"));

    newton_iterations = n_iter;
  }

  void
  debug(VectorType & solution_vector)
  {
    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_vector, "solution");
    data_out.build_patches(1);

    std::ostringstream filename;
    filename << "debug." << counter++ << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);
  }


private:
  NewtonSolverData          solver_data;
  NonlinearOperator &       nonlinear_operator;
  SolverLinearizedProblem & linear_solver;
  DoFHandler<dim> const &   dof_handler;
  mutable int               counter;
  VectorType                residual, increment, tmp;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_H_ */
