/*
 * NewtonSolver.h
 *
 *  Created on: Jun 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_NEWTONSOLVER_H_
#define INCLUDE_NEWTONSOLVER_H_


struct NewtonSolverData
{
  double abs_tol;
  double rel_tol;
  unsigned int max_iter;
};

template<typename Vector, typename Operator, typename SolverLinearizedProblem>
class NewtonSolver
{
public:
  void initialize(NewtonSolverData solver_data_in,
                  Operator *underlying_operator_in,
                  SolverLinearizedProblem *linear_solver_in)
  {
    solver_data = solver_data_in;
    underlying_operator = underlying_operator_in;
    linear_solver = linear_solver_in;

    underlying_operator->initialize_vector_for_newton_solver(residual);
    underlying_operator->initialize_vector_for_newton_solver(increment);
  }

  void solve(Vector &dst, unsigned int &newton_iterations, double &average_linear_iterations)
  {
    // evaluate residual using the given estimate of the solution
    underlying_operator->evaluate_nonlinear_residual(residual,dst);

    double norm_r = residual.l2_norm();
    double norm_r_0 = norm_r;

    // reset average_linear_iterations
    average_linear_iterations = 0.0;

    // Newton iteration
    unsigned int n_iter = 0;
    while(norm_r > this->solver_data.abs_tol && norm_r/norm_r_0 > solver_data.rel_tol && n_iter < solver_data.max_iter)
    {
      // reset increment
      increment = 0.0;

      // multiply by -1.0 since the linearized problem is "LinearMatrix * increment = - residual"
      residual *= -1.0;

      // solve linear problem
      unsigned int linear_iterations =  linear_solver->solve(increment, residual, &dst);

      if(true)//(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "  Number of linear solver iterations: " << linear_iterations << std::endl;

      average_linear_iterations += linear_iterations;

      // update solution
      dst.add(1.0, increment);

      // evaluate residual using the new solution
      underlying_operator->evaluate_nonlinear_residual(residual,dst);

      norm_r = residual.l2_norm();

      ++n_iter;
    }

    if(n_iter >= solver_data.max_iter)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout<<"Newton solver failed to solve nonlinear problem to given tolerance. Maximum number of iterations exceeded!" << std::endl;
    }

    newton_iterations = n_iter;
    if(n_iter > 0)
      average_linear_iterations /= n_iter;

    return;
  }

private:
  NewtonSolverData solver_data;
  Operator *underlying_operator;
  SolverLinearizedProblem *linear_solver;
  Vector residual, increment;
};


#endif /* INCLUDE_NEWTONSOLVER_H_ */
