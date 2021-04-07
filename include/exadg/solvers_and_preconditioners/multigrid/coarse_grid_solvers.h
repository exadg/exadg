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

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/solvers_and_preconditioners/utilities/petsc_operation.h>

namespace ExaDG
{
using namespace dealii;

enum class KrylovSolverType
{
  CG,
  GMRES
};

template<typename Operator>
class MGCoarseKrylov
  : public MGCoarseGridBase<LinearAlgebra::distributed::Vector<typename Operator::value_type>>
{
public:
  typedef double NumberAMG;

  typedef typename Operator::value_type MultigridNumber;

  typedef LinearAlgebra::distributed::Vector<MultigridNumber> VectorType;

  typedef LinearAlgebra::distributed::Vector<NumberAMG> VectorTypeAMG;

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

  MGCoarseKrylov(Operator const &       matrix,
                 AdditionalData const & additional_data,
                 MPI_Comm const &       comm)
    : coarse_matrix(matrix), additional_data(additional_data), mpi_comm(comm)
  {
    if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi)
    {
      preconditioner.reset(new JacobiPreconditioner<Operator>(coarse_matrix));
      std::shared_ptr<JacobiPreconditioner<Operator>> jacobi =
        std::dynamic_pointer_cast<JacobiPreconditioner<Operator>>(preconditioner);
      AssertDimension(jacobi->get_size_of_diagonal(), coarse_matrix.m());
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
    {
      preconditioner.reset(new BlockJacobiPreconditioner<Operator>(coarse_matrix));
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG &&
            additional_data.amg_data.amg_type == AMGType::Trilinos)
    {
      preconditioner_amg.reset(
        new PreconditionerTrilinosAMG<Operator, NumberAMG>(matrix, additional_data.amg_data));
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG &&
            additional_data.amg_data.amg_type == AMGType::Boomer)
    {
      preconditioner_amg.reset(
        new PreconditionerBoomerAMG<Operator, NumberAMG>(matrix, additional_data.amg_data));
    }
    else
    {
      AssertThrow(
        additional_data.preconditioner == MultigridCoarseGridPreconditioner::None ||
          additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi ||
          additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi ||
          additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG,
        ExcMessage("Specified preconditioner for PCG coarse grid solver not implemented."));
    }
  }

  virtual ~MGCoarseKrylov()
  {
  }

  void
  update()
  {
    if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::None)
    {
      // do nothing
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi ||
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
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  virtual void
  operator()(unsigned int const, VectorType & dst, VectorType const & src) const
  {
    VectorType r(src);
    if(additional_data.operator_is_singular)
      set_zero_mean_value(r);

    if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG &&
       additional_data.amg_data.amg_type == AMGType::Trilinos)
    {
      // create temporal vectors of type NumberAMG (double)
      VectorTypeAMG dst_tri;
      dst_tri.reinit(dst, false);
      VectorTypeAMG src_tri;
      src_tri.reinit(r, true);
      src_tri.copy_locally_owned_data_from(r);

      std::shared_ptr<PreconditionerTrilinosAMG<Operator, NumberAMG>> coarse_operator =
        std::dynamic_pointer_cast<PreconditionerTrilinosAMG<Operator, NumberAMG>>(
          preconditioner_amg);

      ReductionControl solver_control(additional_data.solver_data.max_iter,
                                      additional_data.solver_data.abs_tol,
                                      additional_data.solver_data.rel_tol);

      if(additional_data.solver_type == KrylovSolverType::CG)
      {
        SolverCG<VectorTypeAMG> solver(solver_control);
        solver.solve(coarse_operator->system_matrix, dst_tri, src_tri, *preconditioner_amg);
      }
      else if(additional_data.solver_type == KrylovSolverType::GMRES)
      {
        typename SolverGMRES<VectorTypeAMG>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors     = additional_data.solver_data.max_krylov_size;
        gmres_data.right_preconditioning = true;

        SolverGMRES<VectorTypeAMG> solver(solver_control, gmres_data);
        solver.solve(coarse_operator->system_matrix, dst_tri, src_tri, *preconditioner_amg);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }

      // convert NumberAMG (double) -> MultigridNumber (float)
      dst.copy_locally_owned_data_from(dst_tri);
    }
    else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::AMG &&
            additional_data.amg_data.amg_type == AMGType::Boomer)
    {
#ifdef DEAL_II_WITH_PETSC
      apply_petsc_operation(
        dst,
        src,
        [&](PETScWrappers::VectorBase & petsc_dst, PETScWrappers::VectorBase const & petsc_src) {
          std::shared_ptr<PreconditionerBoomerAMG<Operator, NumberAMG>> coarse_operator =
            std::dynamic_pointer_cast<PreconditionerBoomerAMG<Operator, NumberAMG>>(
              preconditioner_amg);

          ReductionControl solver_control(additional_data.solver_data.max_iter,
                                          additional_data.solver_data.abs_tol,
                                          additional_data.solver_data.rel_tol);

          if(additional_data.solver_type == KrylovSolverType::CG)
          {
            PETScWrappers::SolverCG solver(solver_control);
            solver.solve(coarse_operator->system_matrix,
                         petsc_dst,
                         petsc_src,
                         coarse_operator->amg);
          }
          else if(additional_data.solver_type == KrylovSolverType::GMRES)
          {
            PETScWrappers::SolverGMRES solver(solver_control);
            solver.solve(coarse_operator->system_matrix,
                         petsc_dst,
                         petsc_src,
                         coarse_operator->amg);
          }
          else
          {
            AssertThrow(false, ExcMessage("Not implemented."));
          }
        });
#endif
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
        else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi ||
                additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
        {
          solver_data.use_preconditioner = true;
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }

        solver.reset(
          new Krylov::SolverCG<Operator, PreconditionerBase<MultigridNumber>, VectorType>(
            coarse_matrix, *preconditioner, solver_data));
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
        else if(additional_data.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi ||
                additional_data.preconditioner == MultigridCoarseGridPreconditioner::BlockJacobi)
        {
          solver_data.use_preconditioner = true;
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }

        solver.reset(
          new Krylov::SolverGMRES<Operator, PreconditionerBase<MultigridNumber>, VectorType>(
            coarse_matrix, *preconditioner, solver_data, mpi_comm));
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }

      // Note that the preconditioner has already been updated
      solver->solve(dst, src, false);
    }
  }

private:
  const Operator & coarse_matrix;

  std::shared_ptr<PreconditionerBase<MultigridNumber>> preconditioner;

  // we need a separate object here because the AMG preconditioners need double precision
  std::shared_ptr<PreconditionerBase<NumberAMG>> preconditioner_amg;

  AdditionalData additional_data;

  MPI_Comm const mpi_comm;
};


template<typename Vector, typename InverseOperator>
class MGCoarseChebyshev : public MGCoarseGridBase<Vector>
{
public:
  MGCoarseChebyshev(std::shared_ptr<InverseOperator const> inverse) : inverse_operator(inverse)
  {
  }

  virtual ~MGCoarseChebyshev()
  {
  }

  virtual void
  operator()(unsigned int const level, Vector & dst, const Vector & src) const
  {
    AssertThrow(inverse_operator.get() != 0,
                ExcMessage("MGCoarseChebyshev: inverse_operator is not initialized."));

    AssertThrow(level == 0, ExcNotImplemented());

    inverse_operator->vmult(dst, src);
  }

  std::shared_ptr<InverseOperator const> inverse_operator;
};

template<typename Operator>
class MGCoarseAMG
  : public MGCoarseGridBase<LinearAlgebra::distributed::Vector<typename Operator::value_type>>
{
private:
  typedef double NumberAMG;

  typedef LinearAlgebra::distributed::Vector<NumberAMG> VectorTypeAMG;

  typedef LinearAlgebra::distributed::Vector<typename Operator::value_type> VectorTypeMultigrid;

public:
  MGCoarseAMG(Operator const & op, bool const use_boomer_amg, AMGData data = AMGData())
  {
    if(use_boomer_amg)
      amg_preconditioner.reset(new PreconditionerBoomerAMG<Operator, NumberAMG>(op, data));
    else
      amg_preconditioner.reset(new PreconditionerTrilinosAMG<Operator, NumberAMG>(op, data));
  }

  void
  update()
  {
    amg_preconditioner->update();
  }

  void
  operator()(unsigned int const /*level*/,
             VectorTypeMultigrid &       dst,
             VectorTypeMultigrid const & src) const
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
