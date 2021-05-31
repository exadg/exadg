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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_UTIL_COMPUTE_EIGENVALUES_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_UTIL_COMPUTE_EIGENVALUES_H_

namespace ExaDG
{
using namespace dealii;

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
  for(unsigned int i = 0; i < rhs.locally_owned_size(); ++i)
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

} // namespace ExaDG


#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_UTIL_COMPUTE_EIGENVALUES_H_ */
