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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

// deal.II
#include <deal.II/base/function_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer.h>
#include <exadg/utilities/timer_tree.h>

/*
 * Activate timings if desired.
 */
#define ENABLE_TIMING false

#ifndef ENABLE_TIMING
#  define ENABLE_TIMING false
#endif

namespace ExaDG
{
/*
 * Re-implementation of multigrid preconditioner (V-cycle) in order to have more direct control over
 * its individual components and avoid inner products and other expensive stuff.
 */
template<typename VectorType, typename MatrixType, typename SmootherType>
class MultigridAlgorithm
{
public:
  MultigridAlgorithm(dealii::MGLevelObject<std::shared_ptr<MatrixType>> const &   matrix,
                     dealii::MGCoarseGridBase<VectorType> const &                 coarse,
                     MGTransfer<VectorType> const &                               transfer,
                     dealii::MGLevelObject<std::shared_ptr<SmootherType>> const & smoother,
                     MPI_Comm const &                                             comm,
                     unsigned int const                                           n_cycles = 1)
    : minlevel(matrix.min_level()),
      maxlevel(matrix.max_level()),
      defect(minlevel, maxlevel),
      solution(minlevel, maxlevel),
      t(minlevel, maxlevel),
      matrix(&matrix, typeid(*this).name()),
      coarse(&coarse, typeid(*this).name()),
      transfer(transfer),
      smoother(&smoother, typeid(*this).name()),
      mpi_comm(comm),
      n_cycles(n_cycles)
  {
    AssertThrow(n_cycles == 1, dealii::ExcNotImplemented());

    for(unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      matrix[level]->initialize_dof_vector(solution[level]);
      defect[level] = solution[level];
      t[level]      = solution[level];
    }

    timer_tree = std::make_shared<TimerTree>();
  }

  template<class OtherVectorType>
  void
  vmult(OtherVectorType & dst, OtherVectorType const & src) const
  {
#if ENABLE_TIMING
    dealii::Timer timer;
#endif

    for(unsigned int i = minlevel; i < maxlevel; i++)
    {
      defect[i] = 0.0;
    }
    defect[maxlevel].copy_locally_owned_data_from(src);

    v_cycle(maxlevel, false);

    dst.copy_locally_owned_data_from(solution[maxlevel]);

#if ENABLE_TIMING
    timer_tree->insert({"Multigrid"}, timer.wall_time());
#endif
  }

  template<class OtherVectorType>
  unsigned int
  solve(OtherVectorType & dst, OtherVectorType const & src) const
  {
    defect[maxlevel].copy_locally_owned_data_from(src);

    solution[maxlevel].copy_locally_owned_data_from(dst);

    VectorType residual;
    (*matrix)[maxlevel]->initialize_dof_vector(residual);

    // calculate residual and check convergence
    double const norm_r_0 = calculate_residual(residual);
    double       norm_r   = norm_r_0;

    int const    max_iter = 1000;
    double const abstol   = 1.e-12;
    double const reltol   = 1.e-6;

    int  n_iter    = 0;
    bool converged = norm_r_0 < abstol;
    while(not converged)
    {
      for(unsigned int i = minlevel; i < maxlevel; i++)
      {
        defect[i] = 0.0;
      }

      v_cycle(maxlevel, true);

      // calculate residual and check convergence
      norm_r = calculate_residual(residual);
      std::cout << "Norm of residual = " << norm_r << std::endl;
      converged = (norm_r < abstol or norm_r / norm_r_0 < reltol or n_iter >= max_iter);

      ++n_iter;
    }

    dst.copy_locally_owned_data_from(solution[maxlevel]);

    return n_iter;
  }

  template<class OtherVectorType>
  double
  calculate_residual(OtherVectorType & residual) const
  {
    (*matrix)[maxlevel]->vmult(residual, solution[maxlevel]);
    residual.sadd(-1.0, 1.0, defect[maxlevel]);
    return residual.l2_norm();
  }

  std::shared_ptr<TimerTree>
  get_timings() const
  {
    return timer_tree;
  }

private:
  /**
   * Implements the V-cycle
   */
  void
  v_cycle(unsigned int const level, bool const multigrid_is_a_solver) const
  {
#if ENABLE_TIMING
    dealii::Timer timer;
#endif

    // call coarse grid solver
    if(level == minlevel)
    {
#if ENABLE_TIMING
      timer.restart();
#endif

      (*coarse)(level, solution[level], defect[level]);

#if ENABLE_TIMING
      timer_tree->insert({"Multigrid", "level " + std::to_string(level)}, timer.wall_time());
#endif
    }
    else
    {
#if ENABLE_TIMING
      timer.restart();
#endif

      // pre-smoothing
      if(multigrid_is_a_solver)
      {
        // One has to take into account the initial guess of the solution when used as a solver
        // and, therefore, call the function step().
        (*smoother)[level]->step(solution[level], defect[level]);
      }
      else
      {
        // We can assume that solution[level] = 0 when used as a preconditioner
        // and, therefore, call the function vmult(), which makes use of this assumption
        // in order to apply optimizations (e.g., one does not need to evaluate the residual in
        // the first iteration of the smoother).
        (*smoother)[level]->vmult(solution[level], defect[level]);
      }

      // restriction
      (*matrix)[level]->vmult_interface_down(t[level], solution[level]);
      t[level].sadd(-1.0, 1.0, defect[level]);
      transfer.restrict_and_add(level, defect[level - 1], t[level]);

#if ENABLE_TIMING
      timer_tree->insert({"Multigrid", "level " + std::to_string(level)}, timer.wall_time());
#endif

      // coarse grid correction
      v_cycle(level - 1, false);

#if ENABLE_TIMING
      timer.restart();
#endif

      // prolongation
      transfer.prolongate_and_add(level, solution[level], solution[level - 1]);

      // post-smoothing
      (*smoother)[level]->step(solution[level], defect[level]);

#if ENABLE_TIMING
      timer_tree->insert({"Multigrid", "level " + std::to_string(level)}, timer.wall_time());
#endif
    }
  }

  /**
   * Coarsest level.
   */
  unsigned int minlevel;

  /**
   * Finest level.
   */
  unsigned int maxlevel;

  /**
   * Input vector for the cycle. Contains the defect of the outer method
   * projected to the multilevel vectors.
   */
  mutable dealii::MGLevelObject<VectorType> defect;

  /**
   * The solution update after the multigrid step.
   */
  mutable dealii::MGLevelObject<VectorType> solution;

  /**
   * Auxiliary vector.
   */
  mutable dealii::MGLevelObject<VectorType> t;

  /**
   * The matrix for each level.
   */
  dealii::SmartPointer<dealii::MGLevelObject<std::shared_ptr<MatrixType>> const> matrix;

  /**
   * The matrix for each level.
   */
  dealii::SmartPointer<dealii::MGCoarseGridBase<VectorType> const> coarse;

  /**
   * Object for grid transfer.
   */
  MGTransfer<VectorType> const & transfer;

  /**
   * The smoothing object.
   */
  dealii::SmartPointer<dealii::MGLevelObject<std::shared_ptr<SmootherType>> const> smoother;

  MPI_Comm const mpi_comm;

  unsigned int const n_cycles;

  std::shared_ptr<TimerTree> timer_tree;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ */
