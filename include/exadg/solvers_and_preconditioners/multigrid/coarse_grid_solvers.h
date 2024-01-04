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
#include <deal.II/lac/petsc_solver.h>
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
#include <exadg/solvers_and_preconditioners/utilities/petsc_operation.h>

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

enum class KrylovSolverType
{
  CG,
  GMRES
};

template<typename Operator>
class MGCoarseKrylov : public CoarseGridSolverBase<Operator>
{
public:
  typedef double NumberAMG;

  typedef typename Operator::value_type MultigridNumber;

  typedef dealii::LinearAlgebra::distributed::Vector<MultigridNumber> VectorType;

  typedef dealii::LinearAlgebra::distributed::Vector<NumberAMG> VectorTypeAMG;

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
      : solver_type(KrylovSolverType::CG),
        solver_data(SolverData(1e4, 1.e-12, 1.e-3, 100)),
        operator_is_singular(false),
        preconditioner(MultigridCoarseGridPreconditioner::None),
        amg_data(AMGData())
    {
    }

    // Type of Krylov solver
    KrylovSolverType solver_type;

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
      if(additional_data.amg_data.amg_type == AMGType::ML)
      {
#ifdef DEAL_II_WITH_TRILINOS
        preconditioner_amg =
          std::make_shared<PreconditionerML<Operator, NumberAMG>>(pde_operator,
                                                                  initialize,
                                                                  additional_data.amg_data.ml_data);
#else
        AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
      }
      else if(additional_data.amg_data.amg_type == AMGType::BoomerAMG)
      {
#ifdef DEAL_II_WITH_PETSC
        preconditioner_amg = std::make_shared<PreconditionerBoomerAMG<Operator, NumberAMG>>(
          pde_operator, initialize, additional_data.amg_data.boomer_data);
#else
        AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
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
            additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
    {
      preconditioner->update();
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG)
    {
      preconditioner_amg->update();
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
      if(additional_data.amg_data.amg_type == AMGType::ML)
      {
#ifdef DEAL_II_WITH_TRILINOS
        // create temporal vectors of type NumberAMG (double)
        VectorTypeAMG dst_tri;
        dst_tri.reinit(dst, false);
        VectorTypeAMG src_tri;
        src_tri.reinit(r, true);
        src_tri.copy_locally_owned_data_from(r);

        std::shared_ptr<PreconditionerML<Operator, NumberAMG>> coarse_operator =
          std::dynamic_pointer_cast<PreconditionerML<Operator, NumberAMG>>(preconditioner_amg);

        dealii::ReductionControl solver_control(additional_data.solver_data.max_iter,
                                                additional_data.solver_data.abs_tol,
                                                additional_data.solver_data.rel_tol);

        if(additional_data.solver_type == KrylovSolverType::CG)
        {
          dealii::SolverCG<VectorTypeAMG> solver(solver_control);
          solver.solve(coarse_operator->system_matrix, dst_tri, src_tri, *preconditioner_amg);
        }
        else if(additional_data.solver_type == KrylovSolverType::GMRES)
        {
          typename dealii::SolverGMRES<VectorTypeAMG>::AdditionalData gmres_data;
          gmres_data.max_n_tmp_vectors     = additional_data.solver_data.max_krylov_size;
          gmres_data.right_preconditioning = true;

          dealii::SolverGMRES<VectorTypeAMG> solver(solver_control, gmres_data);
          solver.solve(coarse_operator->system_matrix, dst_tri, src_tri, *preconditioner_amg);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        // convert NumberAMG (double) -> MultigridNumber (float)
        dst.copy_locally_owned_data_from(dst_tri);
#endif
      }
      else if(additional_data.amg_data.amg_type == AMGType::BoomerAMG)
      {
#ifdef DEAL_II_WITH_PETSC
        apply_petsc_operation(
          dst,
          src,
          std::dynamic_pointer_cast<PreconditionerBoomerAMG<Operator, NumberAMG>>(
            preconditioner_amg)
            ->system_matrix.get_mpi_communicator(),
          [&](dealii::PETScWrappers::VectorBase &       petsc_dst,
              dealii::PETScWrappers::VectorBase const & petsc_src) {
            std::shared_ptr<PreconditionerBoomerAMG<Operator, NumberAMG>> coarse_operator =
              std::dynamic_pointer_cast<PreconditionerBoomerAMG<Operator, NumberAMG>>(
                preconditioner_amg);

            dealii::ReductionControl solver_control(additional_data.solver_data.max_iter,
                                                    additional_data.solver_data.abs_tol,
                                                    additional_data.solver_data.rel_tol);

            if(additional_data.solver_type == KrylovSolverType::CG)
            {
              dealii::PETScWrappers::SolverCG solver(solver_control);
              solver.solve(coarse_operator->system_matrix,
                           petsc_dst,
                           petsc_src,
                           coarse_operator->amg);
            }
            else if(additional_data.solver_type == KrylovSolverType::GMRES)
            {
              dealii::PETScWrappers::SolverGMRES solver(solver_control);
              solver.solve(coarse_operator->system_matrix,
                           petsc_dst,
                           petsc_src,
                           coarse_operator->amg);
            }
            else
            {
              AssertThrow(false, dealii::ExcMessage("Not implemented."));
            }
          });
#endif
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
    }
    else
    {
      std::shared_ptr<Krylov::SolverBase<VectorType>> solver;

      if(additional_data.solver_type == KrylovSolverType::CG)
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

        solver.reset(
          new Krylov::SolverCG<Operator, PreconditionerBase<MultigridNumber>, VectorType>(
            pde_operator, *preconditioner, solver_data));
      }
      else if(additional_data.solver_type == KrylovSolverType::GMRES)
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

        solver.reset(
          new Krylov::SolverGMRES<Operator, PreconditionerBase<MultigridNumber>, VectorType>(
            pde_operator, *preconditioner, solver_data, mpi_comm));
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }

      // Note that the preconditioner has already been updated
      solver->solve(dst, src);
    }
  }

private:
  const Operator & pde_operator;

  std::shared_ptr<PreconditionerBase<MultigridNumber>> preconditioner;

  // we need a separate object here because the AMG preconditioners need double precision
  std::shared_ptr<PreconditionerBase<NumberAMG>> preconditioner_amg;

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

template<typename Operator>
class MGCoarseAMG : public CoarseGridSolverBase<Operator>
{
private:
  typedef double NumberAMG;

  typedef dealii::LinearAlgebra::distributed::Vector<NumberAMG> VectorTypeAMG;

  typedef dealii::LinearAlgebra::distributed::Vector<typename Operator::value_type>
    VectorTypeMultigrid;

public:
  MGCoarseAMG(Operator const & op, bool const initialize, AMGData data = AMGData())
  {
    (void)op;
    (void)data;

    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      amg_preconditioner =
        std::make_shared<PreconditionerBoomerAMG<Operator, NumberAMG>>(op,
                                                                       initialize,
                                                                       data.boomer_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      amg_preconditioner =
        std::make_shared<PreconditionerML<Operator, NumberAMG>>(op, initialize, data.ml_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  update() final
  {
    amg_preconditioner->update();
  }

  void
  operator()(unsigned int const /*level*/,
             VectorTypeMultigrid &       dst,
             VectorTypeMultigrid const & src) const final
  {
    // create temporal vectors of type VectorTypeAMG (double)
    VectorTypeAMG dst_amg;
    dst_amg.reinit(dst, false);
    VectorTypeAMG src_amg;
    src_amg.reinit(src, true);

    // convert: VectorTypeMultigrid -> VectorTypeAMG
    src_amg.copy_locally_owned_data_from(src);

    // apply AMG as solver
    amg_preconditioner->vmult(dst_amg, src_amg);

    // convert: VectorTypeAMG -> VectorTypeMultigrid
    dst.copy_locally_owned_data_from(dst_amg);
  }

private:
  std::shared_ptr<PreconditionerBase<NumberAMG>> amg_preconditioner;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_ */
