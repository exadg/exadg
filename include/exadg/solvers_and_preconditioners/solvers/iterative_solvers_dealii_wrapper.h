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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

namespace ExaDG
{
using namespace dealii;

template<typename VectorType>
class IterativeSolverBase
{
public:
  IterativeSolverBase() : l2_0(1.0), l2_n(1.0), n(0), rho(0.0), n10(0)
  {
  }

  virtual unsigned int
  solve(VectorType & dst, VectorType const & rhs, bool const update_preconditioner) const = 0;

  virtual ~IterativeSolverBase()
  {
  }

  template<typename Control>
  void
  compute_performance_metrics(Control const & solver_control) const
  {
    // get some statistics related to convergence
    this->l2_0 = solver_control.initial_value();
    this->l2_n = solver_control.last_value();
    this->n    = solver_control.last_step();

    // compute some derived performance metrics
    if(n > 0)
    {
      this->rho = std::pow(l2_n / l2_0, 1.0 / n);
      this->n10 = -10.0 * std::log(10.0) / std::log(rho);
    }
  }

  // performance metrics
  mutable double       l2_0; // norm of initial residual
  mutable double       l2_n; // norm of final residual
  mutable unsigned int n;    // number of iterations
  mutable double       rho;  // average convergence rate
  mutable double       n10;  // number of iterations needed to reduce the residual by 1e10
};

struct CGSolverData
{
  CGSolverData()
    : max_iter(1e4),
      solver_tolerance_abs(1.e-20),
      solver_tolerance_rel(1.e-6),
      use_preconditioner(false),
      compute_performance_metrics(false)
  {
  }

  unsigned int max_iter;
  double       solver_tolerance_abs;
  double       solver_tolerance_rel;
  bool         use_preconditioner;
  bool         compute_performance_metrics;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class CGSolver : public IterativeSolverBase<VectorType>
{
public:
  CGSolver(Operator const &     underlying_operator_in,
           Preconditioner &     preconditioner_in,
           CGSolverData const & solver_data_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in)
  {
  }

  unsigned int
  solve(VectorType & dst, VectorType const & rhs, bool const update_preconditioner) const
  {
    ReductionControl solver_control(solver_data.max_iter,
                                    solver_data.solver_tolerance_abs,
                                    solver_data.solver_tolerance_rel);

    SolverCG<VectorType> solver(solver_control);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      if(update_preconditioner == true)
      {
        preconditioner.update();
      }

      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    if(solver_data.compute_performance_metrics)
      this->compute_performance_metrics(solver_control);

    return solver_control.last_step();
  }

private:
  Operator const &   underlying_operator;
  Preconditioner &   preconditioner;
  CGSolverData const solver_data;
};

template<class NUMBER>
void
output_eigenvalues(const std::vector<NUMBER> & eigenvalues,
                   const std::string &         text,
                   MPI_Comm const &            mpi_comm)
{
  if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::cout << text << std::endl;
    for(unsigned int j = 0; j < eigenvalues.size(); ++j)
    {
      std::cout << ' ' << eigenvalues.at(j) << std::endl;
    }
    std::cout << std::endl;
  }
}

struct GMRESSolverData
{
  GMRESSolverData()
    : max_iter(1e4),
      solver_tolerance_abs(1.e-20),
      solver_tolerance_rel(1.e-6),
      use_preconditioner(false),
      max_n_tmp_vectors(30),
      compute_eigenvalues(false),
      compute_performance_metrics(false)
  {
  }

  unsigned int max_iter;
  double       solver_tolerance_abs;
  double       solver_tolerance_rel;
  bool         use_preconditioner;
  unsigned int max_n_tmp_vectors;
  bool         compute_eigenvalues;
  bool         compute_performance_metrics;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class GMRESSolver : public IterativeSolverBase<VectorType>
{
public:
  GMRESSolver(Operator const &        underlying_operator_in,
              Preconditioner &        preconditioner_in,
              GMRESSolverData const & solver_data_in,
              MPI_Comm const &        mpi_comm_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in),
      mpi_comm(mpi_comm_in)
  {
  }

  virtual ~GMRESSolver()
  {
  }

  unsigned int
  solve(VectorType & dst, VectorType const & rhs, bool const update_preconditioner) const
  {
    ReductionControl solver_control(solver_data.max_iter,
                                    solver_data.solver_tolerance_abs,
                                    solver_data.solver_tolerance_rel);

    typename SolverGMRES<VectorType>::AdditionalData additional_data;
    additional_data.max_n_tmp_vectors     = solver_data.max_n_tmp_vectors;
    additional_data.right_preconditioning = true;
    SolverGMRES<VectorType> solver(solver_control, additional_data);

    if(solver_data.compute_eigenvalues == true)
    {
      solver.connect_eigenvalues_slot(std::bind(output_eigenvalues<std::complex<double>>,
                                                std::placeholders::_1,
                                                "Eigenvalues: ",
                                                mpi_comm),
                                      true);
    }

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      if(update_preconditioner == true)
      {
        preconditioner.update();
      }

      solver.solve(this->underlying_operator, dst, rhs, this->preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    if(solver_data.compute_performance_metrics)
      this->compute_performance_metrics(solver_control);

    return solver_control.last_step();
  }

private:
  Operator const &      underlying_operator;
  Preconditioner &      preconditioner;
  GMRESSolverData const solver_data;

  MPI_Comm const mpi_comm;
};

struct FGMRESSolverData
{
  FGMRESSolverData()
    : max_iter(1e4),
      solver_tolerance_abs(1.e-20),
      solver_tolerance_rel(1.e-6),
      use_preconditioner(false),
      max_n_tmp_vectors(30),
      compute_performance_metrics(false)
  {
  }

  unsigned int max_iter;
  double       solver_tolerance_abs;
  double       solver_tolerance_rel;
  bool         use_preconditioner;
  unsigned int max_n_tmp_vectors;
  bool         compute_performance_metrics;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class FGMRESSolver : public IterativeSolverBase<VectorType>
{
public:
  FGMRESSolver(Operator const &         underlying_operator_in,
               Preconditioner &         preconditioner_in,
               FGMRESSolverData const & solver_data_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in)
  {
  }

  virtual ~FGMRESSolver()
  {
  }

  unsigned int
  solve(VectorType & dst, VectorType const & rhs, bool const update_preconditioner) const
  {
    ReductionControl solver_control(solver_data.max_iter,
                                    solver_data.solver_tolerance_abs,
                                    solver_data.solver_tolerance_rel);

    typename SolverFGMRES<VectorType>::AdditionalData additional_data;
    additional_data.max_basis_size = solver_data.max_n_tmp_vectors;
    // FGMRES always uses right preconditioning

    SolverFGMRES<VectorType> solver(solver_control, additional_data);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      if(update_preconditioner == true)
      {
        preconditioner.update();
      }

      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    if(solver_data.compute_performance_metrics)
      this->compute_performance_metrics(solver_control);

    return solver_control.last_step();
  }

private:
  Operator const &       underlying_operator;
  Preconditioner &       preconditioner;
  FGMRESSolverData const solver_data;
};
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_ */
