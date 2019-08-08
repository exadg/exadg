/*
 * iterative_solvers_dealii_wrapper.h
 *
 *  Created on: Aug 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

template<typename VectorType>
class IterativeSolverBase
{
public:
  IterativeSolverBase()
    : performance_metrics_available(false), l2_0(1.0), l2_n(1.0), n(0), rho(1.0), r(1.0), n10(0)
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
    performance_metrics_available = true;

    // get some statistics related to convergence
    this->l2_0 = solver_control.initial_value();
    this->l2_n = solver_control.last_value();
    this->n    = solver_control.last_step();

    // compute some derived performance metrics
    AssertThrow(n != 0, ExcMessage("Division by zero."));
    this->rho = std::pow(l2_n / l2_0, 1.0 / n);
    this->r   = -std::log(rho) / std::log(10.0);
    this->n10 = -10.0 * std::log(10.0) / std::log(rho);
  }

  // performance metrics
  mutable bool         performance_metrics_available;
  mutable double       l2_0; // norm of initial residual
  mutable double       l2_n; // norm of final residual
  mutable unsigned int n;    // number of iterations
  mutable double       rho;  // average convergence rate
  mutable double       r;    // logarithmic convergence rate
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
output_eigenvalues(const std::vector<NUMBER> & eigenvalues, const std::string & text)
{
  //    deallog << text << std::endl;
  //    for (unsigned int j = 0; j < eigenvalues.size(); ++j)
  //      {
  //        deallog << ' ' << eigenvalues.at(j) << std::endl;
  //      }
  //    deallog << std::endl;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
              GMRESSolverData const & solver_data_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in)
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
                                                "Eigenvalues: "),
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

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_ */
