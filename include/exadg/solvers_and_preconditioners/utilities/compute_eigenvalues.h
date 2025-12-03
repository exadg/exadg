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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_COMPUTE_EIGENVALUES_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_COMPUTE_EIGENVALUES_H_

// deal.II
#include <deal.II/numerics/vector_tools_mean_value.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>

namespace ExaDG
{
/*
 * Utility function to estimate the Eigenvalues of `Operator` via a
 * conjugate gradient solve preconditioned by the inverse diagonal.
 */
template<typename Operator, typename VectorType>
std::pair<double, double>
estimate_eigenvalues_cg(Operator const &   underlying_operator,
                        VectorType const & inverse_diagonal,
                        bool const         operator_is_singular,
                        unsigned int const eig_n_iter = 10000)
{
  VectorType solution, rhs;
  solution.reinit(inverse_diagonal);
  rhs.reinit(inverse_diagonal, true);
  // seed `rand` in order to obtain reproducible results
  srand(1);
  for(unsigned int i = 0; i < rhs.locally_owned_size(); ++i)
    rhs.local_element(i) = (double)rand() / RAND_MAX;
  if(operator_is_singular)
    dealii::VectorTools::subtract_mean_value(rhs);

  dealii::SolverControl               control(eig_n_iter, rhs.l2_norm() * 1e-5);
  dealii::internal::EigenvalueTracker eigenvalue_tracker;

  dealii::SolverCG<VectorType> solver(control);

  solver.connect_eigenvalues_slot(std::bind(&dealii::internal::EigenvalueTracker::slot,
                                            &eigenvalue_tracker,
                                            std::placeholders::_1),
                                  false /* every_iteration */);

  JacobiPreconditioner<Operator> preconditioner(underlying_operator, true /* initialize */);

  try
  {
    solver.solve(underlying_operator, solution, rhs, preconditioner);
  }
  catch(dealii::SolverControl::NoConvergence &)
  {
    // accept the current estimates despite non-convergence
  }

  std::pair<double, double> eigenvalues_min_max;
  if(eigenvalue_tracker.values.empty())
  {
    eigenvalues_min_max.first = eigenvalues_min_max.second = 1.;
  }
  else
  {
    eigenvalues_min_max.first  = eigenvalue_tracker.values.front();
    eigenvalues_min_max.second = eigenvalue_tracker.values.back();
  }

  return eigenvalues_min_max;
}

// Struct to track complex eigenvalues similar to `dealii::internals::EigenValueTracker`.
struct EigenvalueTracker
{
public:
  void
  slot(std::vector<std::complex<double>> const & eigenvalues)
  {
    values = eigenvalues;
  }

  std::vector<std::complex<double>> values;
};

/*
 * Utility function to estimate and print the Eigenvalues of `Operator` via a preconditioned GMRES
 * solve. This function is used for debugging only and assumes a previous solve such that `solution`
 * and `rhs` solve the system up to tolerance already.
 */
template<typename Operator, typename Preconditioner, typename VectorType>
std::vector<std::complex<double>>
estimate_eigenvalues_gmres(Operator const &                underlying_operator,
                           Preconditioner const &          preconditioner,
                           VectorType const &              solution,
                           VectorType const &              rhs,
                           Krylov::SolverDataGMRES const & solver_data,
                           bool const                      print)
{
  // initial guess
  VectorType tmp(solution);

  dealii::ReductionControl solver_control(solver_data.max_iter,
                                          solver_data.solver_tolerance_abs,
                                          solver_data.solver_tolerance_rel);

  typename dealii::SolverGMRES<VectorType>::AdditionalData additional_data;
  additional_data.max_n_tmp_vectors     = solver_data.max_n_tmp_vectors;
  additional_data.right_preconditioning = true;
  dealii::SolverGMRES<VectorType> solver(solver_control, additional_data);

  EigenvalueTracker eigenvalue_tracker;
  solver.connect_eigenvalues_slot(std::bind(&EigenvalueTracker::slot,
                                            &eigenvalue_tracker,
                                            std::placeholders::_1),
                                  true /* every_iteration */);

  try
  {
    solver.solve(underlying_operator, tmp, rhs, preconditioner);
  }
  catch(dealii::SolverControl::NoConvergence &)
  {
    // accept the current estimates despite non-convergence
  }

  std::vector<std::complex<double>> const & eigenvalues = eigenvalue_tracker.values;

  if(print)
  {
    MPI_Comm const & mpi_comm = solution.get_mpi_communicator();
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::cout << "Approximate eigenvalues:\n";
      for(unsigned int j = 0; j < eigenvalues.size(); ++j)
      {
        std::cout << ' ' << eigenvalues[j] << "\n";
      }
      std::cout << "\n";
    }
  }

  return eigenvalues;
}
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_COMPUTE_EIGENVALUES_H_ */
