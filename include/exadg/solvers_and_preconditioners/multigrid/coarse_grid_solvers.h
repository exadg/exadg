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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/chebyshev_smoother.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/solvers_and_preconditioners/utilities/linear_algebra_utilities.h>

namespace ExaDG
{
/**
 * Base class for multigrid coarse-grid solvers in order to define update() function in addition to
 * the interface of dealii::MGCoarseGridBase.
 */
template<typename Operator>
class CoarseGridSolverBase
  : public dealii::MGCoarseGridBase<
      dealii::LinearAlgebra::distributed::Vector<typename Operator::value_type>>
{
public:
  virtual ~CoarseGridSolverBase(){};

  virtual void
  update() = 0;
};

template<typename Operator>
class MGCoarseKrylov : public CoarseGridSolverBase<Operator>
{
public:
  typedef typename Operator::value_type Number;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
      : solver_type(MultigridCoarseGridSolver::CG),
        solver_data(SolverData(1e4, 1.e-12, 1.e-3, 100)),
        operator_is_singular(false),
        preconditioner(MultigridCoarseGridPreconditioner::None),
        amg_data(AMGData())
    {
    }

    // Type of Krylov solver
    MultigridCoarseGridSolver solver_type;

    // Solver data
    SolverData solver_data;

    // in case of singular operators (with constant vectors forming the nullspace) the rhs vector
    // has to be projected onto the space of vectors with zero mean prior to solving the coarse
    // grid problem
    bool operator_is_singular;

    // Preconditioner
    MultigridCoarseGridPreconditioner preconditioner;

    // Configuration of AMG settings
    AMGData amg_data;
  };

  MGCoarseKrylov(Operator const &       pde_operator_in,
                 bool const             initialize,
                 AdditionalData const & additional_data,
                 MPI_Comm const &       comm)
    : pde_operator(pde_operator_in), additional_data(additional_data), mpi_comm(comm)
  {
    if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi)
    {
      preconditioner = std::make_shared<JacobiPreconditioner<Operator>>(pde_operator, initialize);

      std::shared_ptr<JacobiPreconditioner<Operator>> jacobi =
        std::dynamic_pointer_cast<JacobiPreconditioner<Operator>>(preconditioner);
      AssertDimension(jacobi->get_size_of_diagonal(), pde_operator.m());
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
    {
      preconditioner =
        std::make_shared<BlockJacobiPreconditioner<Operator>>(pde_operator, initialize);
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG)
    {
      preconditioner =
        std::make_shared<PreconditionerAMG<Operator, Number>>(pde_operator,
                                                              initialize,
                                                              additional_data.amg_data);
    }
    else
    {
      AssertThrow(
        additional_data.preconditioner == MultigridCoarseGridPreconditioner::None or
          additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi or
          additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi or
          additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG,
        dealii::ExcMessage("Specified preconditioner for PCG coarse grid solver not implemented."));
    }
  }

  void
  update() final
  {
    if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::None)
    {
      // do nothing
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi or
            additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi or
            additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG)
    {
      preconditioner->update();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  void
  operator()(unsigned int const, VectorType & dst, VectorType const & src) const final
  {
    VectorType r(src);
    if(additional_data.operator_is_singular)
      dealii::VectorTools::subtract_mean_value(r);

    if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG)
    {
      std::shared_ptr<PreconditionerAMG<Operator, Number>> preconditioner_amg =
        std::dynamic_pointer_cast<PreconditionerAMG<Operator, Number>>(preconditioner);

      preconditioner_amg->apply_krylov_solver_with_amg_preconditioner(dst,
                                                                      r,
                                                                      additional_data.solver_type,
                                                                      additional_data.solver_data);
    }
    else
    {
      std::shared_ptr<Krylov::SolverBase<VectorType>> solver;

      if(additional_data.solver_type == MultigridCoarseGridSolver::CG)
      {
        Krylov::SolverDataCG solver_data;
        solver_data.max_iter             = additional_data.solver_data.max_iter;
        solver_data.solver_tolerance_abs = additional_data.solver_data.abs_tol;
        solver_data.solver_tolerance_rel = additional_data.solver_data.rel_tol;

        if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::None)
        {
          solver_data.use_preconditioner = false;
        }
        else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi or
                additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
        {
          solver_data.use_preconditioner = true;
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        solver.reset(new Krylov::SolverCG<Operator, PreconditionerBase<Number>, VectorType>(
          pde_operator, *preconditioner, solver_data));
      }
      else if(additional_data.solver_type == MultigridCoarseGridSolver::GMRES)
      {
        Krylov::SolverDataGMRES solver_data;

        solver_data.max_iter             = additional_data.solver_data.max_iter;
        solver_data.solver_tolerance_abs = additional_data.solver_data.abs_tol;
        solver_data.solver_tolerance_rel = additional_data.solver_data.rel_tol;
        solver_data.max_n_tmp_vectors    = additional_data.solver_data.max_krylov_size;

        if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::None)
        {
          solver_data.use_preconditioner = false;
        }
        else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi or
                additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
        {
          solver_data.use_preconditioner = true;
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        solver.reset(new Krylov::SolverGMRES<Operator, PreconditionerBase<Number>, VectorType>(
          pde_operator, *preconditioner, solver_data, mpi_comm));
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }

      // Note that the preconditioner has already been updated
      solver->solve(dst, r);
    }
  }

private:
  const Operator & pde_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  AdditionalData additional_data;

  MPI_Comm const mpi_comm;
};


template<typename Operator>
class MGCoarseChebyshev : public CoarseGridSolverBase<Operator>
{
public:
  typedef typename Operator::value_type MultigridNumber;

  typedef dealii::LinearAlgebra::distributed::Vector<MultigridNumber> VectorType;

  typedef dealii::PreconditionChebyshev<Operator, VectorType, dealii::DiagonalMatrix<VectorType>>
    DealiiChebyshev;

  MGCoarseChebyshev(Operator const &                          coarse_operator_in,
                    bool const                                initialize_preconditioner_in,
                    double const                              relative_tolerance_in,
                    MultigridCoarseGridPreconditioner const & preconditioner,
                    bool const                                operator_is_singular_in)
    : coarse_operator(coarse_operator_in),
      relative_tolerance(relative_tolerance_in),
      operator_is_singular(operator_is_singular_in)
  {
    AssertThrow(preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
                dealii::ExcMessage(
                  "Only PointJacobi preconditioner implemented for Chebyshev coarse-grid solver."));

    if(initialize_preconditioner_in)
    {
      update();
    }
  }

  void
  update() final
  {
    // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
    typename DealiiChebyshev::AdditionalData dealii_additional_data;

    std::shared_ptr<dealii::DiagonalMatrix<VectorType>> diagonal_matrix =
      std::make_shared<dealii::DiagonalMatrix<VectorType>>();
    VectorType & diagonal_vector = diagonal_matrix->get_vector();

    coarse_operator.initialize_dof_vector(diagonal_vector);
    coarse_operator.calculate_inverse_diagonal(diagonal_vector);

    std::pair<double, double> eigenvalues =
      compute_eigenvalues(coarse_operator, diagonal_vector, operator_is_singular);

    double const factor = 1.1;

    dealii_additional_data.preconditioner  = diagonal_matrix;
    dealii_additional_data.max_eigenvalue  = factor * eigenvalues.second;
    dealii_additional_data.smoothing_range = eigenvalues.second / eigenvalues.first * factor;

    double sigma = (1. - std::sqrt(1. / dealii_additional_data.smoothing_range)) /
                   (1. + std::sqrt(1. / dealii_additional_data.smoothing_range));

    // calculate/estimate the number of Chebyshev iterations needed to reach a specified relative
    // solver tolerance
    double const eps = relative_tolerance;

    dealii_additional_data.degree = static_cast<unsigned int>(
      std::log(1. / eps + std::sqrt(1. / eps / eps - 1.)) / std::log(1. / sigma));
    dealii_additional_data.eig_cg_n_iterations = 0;

    chebyshev_smoother = std::make_shared<DealiiChebyshev>();
    chebyshev_smoother->initialize(coarse_operator, dealii_additional_data);
  }

  void
  operator()(unsigned int const level, VectorType & dst, const VectorType & src) const final
  {
    AssertThrow(chebyshev_smoother.get() != 0,
                dealii::ExcMessage("MGCoarseChebyshev: chebyshev_smoother is not initialized."));

    AssertThrow(level == 0, dealii::ExcNotImplemented());

    chebyshev_smoother->vmult(dst, src);
  }

private:
  Operator const & coarse_operator;
  double const     relative_tolerance;
  bool const       operator_is_singular;

  std::shared_ptr<DealiiChebyshev> chebyshev_smoother;
};

/**
 * The aim if this class is to translate PreconditionerAMG to a coarse-grid solver with the function
 * operator()().
 */
template<typename Operator>
class MGCoarseAMG : public CoarseGridSolverBase<Operator>
{
private:
  typedef typename Operator::value_type Number;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  MGCoarseAMG(Operator const & op, bool const initialize, AMGData data = AMGData())
  {
    amg_preconditioner =
      std::make_shared<PreconditionerAMG<Operator, Number>>(op, initialize, data);
  }

  void
  update() final
  {
    amg_preconditioner->update();
  }

  void
  operator()(unsigned int const /*level*/, VectorType & dst, VectorType const & src) const final
  {
    amg_preconditioner->vmult(dst, src);
  }

private:
  std::shared_ptr<PreconditionerAMG<Operator, Number>> amg_preconditioner;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_ */
