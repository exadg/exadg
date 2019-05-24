/*
 * compute_eigenvalues.h
 *
 *  Created on: Nov 28, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_UTIL_COMPUTE_EIGENVALUES_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_UTIL_COMPUTE_EIGENVALUES_H_


namespace
{
// manually compute eigenvalues for the coarsest level for proper setup of the
// Chebyshev iteration
template<typename Operator, typename VectorType>
std::pair<double, double>
compute_eigenvalues(Operator const &   op,
                    VectorType const & inverse_diagonal,
                    bool const         operator_is_singular,
                    unsigned int const eig_n_iter = 10000)
{
  VectorType solution, rhs;
  solution.reinit(inverse_diagonal);
  rhs.reinit(inverse_diagonal, true);
  // NB: initialize rand in order to obtain "reproducible" results !!!
  srand(1);
  for(unsigned int i = 0; i < rhs.local_size(); ++i)
    rhs.local_element(i) = (double)rand() / RAND_MAX;
  if(operator_is_singular)
    set_zero_mean_value(rhs);

  SolverControl control(eig_n_iter, rhs.l2_norm() * 1e-5);
  internal::PreconditionChebyshevImplementation::EigenvalueTracker eigenvalue_tracker;

  SolverCG<VectorType> solver(control);

  solver.connect_eigenvalues_slot(
    std::bind(&internal::PreconditionChebyshevImplementation::EigenvalueTracker::slot,
              &eigenvalue_tracker,
              std::placeholders::_1));

  JacobiPreconditioner<Operator> preconditioner(op);

  try
  {
    solver.solve(op, solution, rhs, preconditioner);
  }
  catch(SolverControl::NoConvergence &)
  {
  }

  std::pair<double, double> eigenvalues;
  if(eigenvalue_tracker.values.empty())
  {
    eigenvalues.first = eigenvalues.second = 1.;
  }
  else
  {
    eigenvalues.first  = eigenvalue_tracker.values.front();
    eigenvalues.second = eigenvalue_tracker.values.back();
  }

  return eigenvalues;
}

template<typename Number>
struct EigenvalueTracker
{
public:
  void
  slot(const std::vector<Number> & eigenvalues)
  {
    values = eigenvalues;
  }

  std::vector<Number> values;
};

// manually compute eigenvalues for the coarsest level for proper setup of the
// Chebyshev iteration
template<typename Operator, typename VectorType>
std::pair<std::complex<double>, std::complex<double>>
compute_eigenvalues_gmres(Operator const &   op,
                          VectorType const & inverse_diagonal,
                          bool const         operator_is_singular,
                          unsigned int const eig_n_iter = 10000)
{
  VectorType solution, rhs;
  solution.reinit(inverse_diagonal);
  rhs.reinit(inverse_diagonal, true);
  // NB: initialize rand in order to obtain "reproducible" results !!!
  srand(1);
  for(unsigned int i = 0; i < rhs.local_size(); ++i)
    rhs.local_element(i) = (double)rand() / RAND_MAX;
  if(operator_is_singular)
    set_zero_mean_value(rhs);

  ReductionControl control(eig_n_iter, rhs.l2_norm() * 1.0e-5, 1.0e-5);

  EigenvalueTracker<std::complex<double>> eigenvalue_tracker;

  SolverGMRES<VectorType> solver(control);

  solver.connect_eigenvalues_slot(std::bind(&EigenvalueTracker<std::complex<double>>::slot,
                                            &eigenvalue_tracker,
                                            std::placeholders::_1));

  JacobiPreconditioner<Operator> preconditioner(op);

  try
  {
    solver.solve(op, solution, rhs, preconditioner);
  }
  catch(SolverControl::NoConvergence &)
  {
  }

  std::pair<std::complex<double>, std::complex<double>> eigenvalues;
  if(eigenvalue_tracker.values.empty())
  {
    eigenvalues.first = eigenvalues.second = 1.;
  }
  else
  {
    eigenvalues.first  = eigenvalue_tracker.values.front();
    eigenvalues.second = eigenvalue_tracker.values.back();
  }
  return eigenvalues;
}
} // namespace


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_UTIL_COMPUTE_EIGENVALUES_H_ */
