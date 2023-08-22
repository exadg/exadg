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

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Krylov
{
template<typename VectorType>
class SolverBase
{
public:
  SolverBase() : l2_0(1.0), l2_n(1.0), n(0), rho(0.0), n10(0)
  {
    timer_tree = std::make_shared<TimerTree>();
  }

  virtual unsigned int
  solve(VectorType & dst, VectorType const & rhs) const = 0;

  virtual ~SolverBase()
  {
  }

  virtual void
  update_preconditioner(bool const update_preconditioner) const = 0;

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

  virtual std::shared_ptr<TimerTree>
  get_timings() const
  {
    return timer_tree;
  }

  // performance metrics
  mutable double       l2_0; // norm of initial residual
  mutable double       l2_n; // norm of final residual
  mutable unsigned int n;    // number of iterations
  mutable double       rho;  // average convergence rate
  mutable double       n10;  // number of iterations needed to reduce the residual by 1e10

protected:
  std::shared_ptr<TimerTree> timer_tree;
};

struct SolverDataCG
{
  SolverDataCG()
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
class SolverCG : public SolverBase<VectorType>
{
public:
  SolverCG(Operator const &     underlying_operator_in,
           Preconditioner &     preconditioner_in,
           SolverDataCG const & solver_data_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in)
  {
  }

  void
  update_preconditioner(bool const update_preconditioner) const override
  {
    if(solver_data.use_preconditioner and update_preconditioner)
    {
      preconditioner.update();
    }
  }

  unsigned int
  solve(VectorType & dst, VectorType const & rhs) const override
  {
    dealii::Timer timer;

    dealii::ReductionControl solver_control(solver_data.max_iter,
                                            solver_data.solver_tolerance_abs,
                                            solver_data.solver_tolerance_rel);

    dealii::SolverCG<VectorType> solver(solver_control);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, dealii::PreconditionIdentity());
    }
    else
    {
      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                dealii::ExcMessage("Solver contained NaN of Inf values"));

    if(solver_data.compute_performance_metrics)
      this->compute_performance_metrics(solver_control);

    this->timer_tree->insert({"SolverCG"}, timer.wall_time());

    return solver_control.last_step();
  }

  std::shared_ptr<TimerTree>
  get_timings() const override
  {
    if(solver_data.use_preconditioner)
    {
      this->timer_tree->insert({"SolverCG"}, preconditioner.get_timings());
    }

    return this->timer_tree;
  }

private:
  Operator const &   underlying_operator;
  Preconditioner &   preconditioner;
  SolverDataCG const solver_data;
};

template<class Number>
void
output_eigenvalues(const std::vector<Number> & eigenvalues,
                   std::string const &         text,
                   MPI_Comm const &            mpi_comm)
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::cout << text << std::endl;
    for(unsigned int j = 0; j < eigenvalues.size(); ++j)
    {
      std::cout << ' ' << eigenvalues.at(j) << std::endl;
    }
    std::cout << std::endl;
  }
}

struct SolverDataGMRES
{
  SolverDataGMRES()
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
class SolverGMRES : public SolverBase<VectorType>
{
public:
  SolverGMRES(Operator const &        underlying_operator_in,
              Preconditioner &        preconditioner_in,
              SolverDataGMRES const & solver_data_in,
              MPI_Comm const &        mpi_comm_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in),
      mpi_comm(mpi_comm_in)
  {
  }

  virtual ~SolverGMRES()
  {
  }

  void
  update_preconditioner(bool const update_preconditioner) const override
  {
    if(solver_data.use_preconditioner and update_preconditioner)
    {
      preconditioner.update();
    }
  }

  unsigned int
  solve(VectorType & dst, VectorType const & rhs) const override
  {
    dealii::Timer timer;

    dealii::ReductionControl solver_control(solver_data.max_iter,
                                            solver_data.solver_tolerance_abs,
                                            solver_data.solver_tolerance_rel);

    typename dealii::SolverGMRES<VectorType>::AdditionalData additional_data;
    additional_data.max_n_tmp_vectors     = solver_data.max_n_tmp_vectors;
    additional_data.right_preconditioning = true;
    dealii::SolverGMRES<VectorType> solver(solver_control, additional_data);

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
      solver.solve(underlying_operator, dst, rhs, dealii::PreconditionIdentity());
    }
    else
    {
      solver.solve(this->underlying_operator, dst, rhs, this->preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                dealii::ExcMessage("Solver contained NaN of Inf values"));

    if(solver_data.compute_performance_metrics)
      this->compute_performance_metrics(solver_control);

    this->timer_tree->insert({"SolverGMRES"}, timer.wall_time());

    return solver_control.last_step();
  }

  std::shared_ptr<TimerTree>
  get_timings() const override
  {
    if(solver_data.use_preconditioner)
    {
      this->timer_tree->insert({"SolverGMRES"}, preconditioner.get_timings());
    }

    return this->timer_tree;
  }

private:
  Operator const &      underlying_operator;
  Preconditioner &      preconditioner;
  SolverDataGMRES const solver_data;

  MPI_Comm const mpi_comm;
};

struct SolverDataFGMRES
{
  SolverDataFGMRES()
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
class SolverFGMRES : public SolverBase<VectorType>
{
public:
  SolverFGMRES(Operator const &         underlying_operator_in,
               Preconditioner &         preconditioner_in,
               SolverDataFGMRES const & solver_data_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      solver_data(solver_data_in)
  {
  }

  virtual ~SolverFGMRES()
  {
  }

  void
  update_preconditioner(bool const update_preconditioner) const override
  {
    if(solver_data.use_preconditioner and update_preconditioner)
    {
      preconditioner.update();
    }
  }

  unsigned int
  solve(VectorType & dst, VectorType const & rhs) const override
  {
    dealii::Timer timer;

    dealii::ReductionControl solver_control(solver_data.max_iter,
                                            solver_data.solver_tolerance_abs,
                                            solver_data.solver_tolerance_rel);

    typename dealii::SolverFGMRES<VectorType>::AdditionalData additional_data;
    additional_data.max_basis_size = solver_data.max_n_tmp_vectors;
    // FGMRES always uses right preconditioning

    dealii::SolverFGMRES<VectorType> solver(solver_control, additional_data);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, dealii::PreconditionIdentity());
    }
    else
    {
      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                dealii::ExcMessage("Solver contained NaN of Inf values"));

    if(solver_data.compute_performance_metrics)
      this->compute_performance_metrics(solver_control);

    this->timer_tree->insert({"SolverFGMRES"}, timer.wall_time());

    return solver_control.last_step();
  }

  std::shared_ptr<TimerTree>
  get_timings() const override
  {
    if(solver_data.use_preconditioner)
    {
      this->timer_tree->insert({"SolverFGMRES"}, preconditioner.get_timings());
    }

    return this->timer_tree;
  }

private:
  Operator const &       underlying_operator;
  Preconditioner &       preconditioner;
  SolverDataFGMRES const solver_data;
};
} // namespace Krylov

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_ */
