/*
 * multigrid_algorithm.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

#include <deal.II/base/function_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include "../../functionalities/dynamic_convergence_table.h"

#include "transfer/mg_transfer_mf.h"

using namespace dealii;

/*
 * Activate timings if desired.
 */

#define ENABLE_TIMING true

#ifndef ENABLE_TIMING
#  define ENABLE_TIMING false
#endif

using namespace dealii;
/*
 * Re-implementation of multigrid preconditioner (V-cycle) in order to have more direct control over
 * its individual components and avoid inner products and other expensive stuff.
 */
template<typename VectorType, typename MatrixType, typename SmootherType>
class MultigridPreconditioner
{
public:
  MultigridPreconditioner(const MGLevelObject<std::shared_ptr<MatrixType>> &   matrix,
                          const MGCoarseGridBase<VectorType> &                 coarse,
                          const MGTransferMF<VectorType> &                     transfer,
                          const MGLevelObject<std::shared_ptr<SmootherType>> & smoother,
                          const unsigned int                                   n_cycles = 1)
    : minlevel(matrix.min_level()),
      maxlevel(matrix.max_level()),
      defect(minlevel, maxlevel),
      solution(minlevel, maxlevel),
      t(minlevel, maxlevel),
      defect2(minlevel, maxlevel),
      matrix(&matrix, typeid(*this).name()),
      coarse(&coarse, typeid(*this).name()),
      transfer(transfer),
      smoother(&smoother, typeid(*this).name()),
      n_cycles(n_cycles)
  {
    AssertThrow(n_cycles == 1, ExcNotImplemented());

    for(unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      matrix[level]->initialize_dof_vector(solution[level]);
      defect[level] = solution[level];
      t[level]      = solution[level];
      if(n_cycles > 1)
        defect2[level] = solution[level];
    }

#if ENABLE_TIMING
    for(unsigned int level = minlevel; level <= maxlevel; ++level)
      this->table.set("dofs-" + std::to_string(level), defect[level].size());
#endif
  }

  virtual ~MultigridPreconditioner()
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      this->table.print();
  }

  template<class OtherVectorType>
  void
  vmult(OtherVectorType & dst, OtherVectorType const & src) const
  {
    for(unsigned int i = minlevel; i <= maxlevel; i++)
    {
      defect[i] = 0.0;
    }
    defect[maxlevel].copy_locally_owned_data_from(src);

    v_cycle(maxlevel, false);

    dst.copy_locally_owned_data_from(solution[maxlevel]);
  }

  template<class OtherVectorType>
  unsigned int
  solve(OtherVectorType & dst, OtherVectorType const & src) const
  {
    defect[maxlevel] = 0.0;
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
    while(!converged)
    {
      for(unsigned int i = minlevel; i < maxlevel; i++)
      {
        defect[i] = 0.0;
      }

      v_cycle(maxlevel, true);

      // calculate residual and check convergence
      norm_r = calculate_residual(residual);
      std::cout << "Norm of residual = " << norm_r << std::endl;
      converged = (norm_r < abstol || norm_r / norm_r_0 < reltol || n_iter >= max_iter);

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

private:
#if ENABLE_TIMING
  void
  put_in_table(std::string label, Timer & timer) const
  {
    this->table.put(label, timer.wall_time());
  }
#else
  void
  put_in_table(std::string /*label*/, Timer & /*timer*/) const
  {
  }
#endif

  /**
   * Implements the V-cycle
   */
  void
  v_cycle(unsigned int const level, bool const multigrid_is_a_solver) const
  {
    Timer timer_local;
    Timer timer_global;
    timer_global.restart();

    // call coarse grid solver
    if(level == minlevel)
    {
      (*coarse)(level, solution[level], defect[level]);
      put_in_table("level-" + std::to_string(level), timer_global);
    }
    else
    {
      timer_local.restart();

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
      (*matrix)[level]->vmult_interface_down(t[level], solution[level]);
      t[level].sadd(-1.0, 1.0, defect[level]);
      put_in_table("pres-" + std::to_string(level), timer_local);

      // restriction
      timer_local.restart();
      transfer.restrict_and_add(level, defect[level - 1], t[level]);
      put_in_table("rest-" + std::to_string(level), timer_local);

      // coarse grid correction
      v_cycle(level - 1, false);

      timer_local.restart();

      // prolongation
      transfer.prolongate(level, t[level], solution[level - 1]);
      put_in_table("prol-" + std::to_string(level), timer_local);
      timer_local.restart();
      solution[level] += t[level];

      // post-smoothing
      (*smoother)[level]->step(solution[level], defect[level]);
      put_in_table("poss-" + std::to_string(level), timer_local);
      put_in_table("level-" + std::to_string(level), timer_global);
    }
  }

  /**
   * Lowest level of cells.
   */
  unsigned int minlevel;

  /**
   * Highest level of cells.
   */
  unsigned int maxlevel;

  /**
   * Input vector for the cycle. Contains the defect of the outer method
   * projected to the multilevel vectors.
   */
  mutable MGLevelObject<VectorType> defect;

  /**
   * The solution update after the multigrid step.
   */
  mutable MGLevelObject<VectorType> solution;

  /**
   * Auxiliary vector.
   */
  mutable MGLevelObject<VectorType> t;

  /**
   * Auxiliary vector if more than 1 cycle is needed
   */
  mutable MGLevelObject<VectorType> defect2;

  /**
   * The matrix for each level.
   */
  SmartPointer<const MGLevelObject<std::shared_ptr<MatrixType>>> matrix;

  /**
   * The matrix for each level.
   */
  SmartPointer<const MGCoarseGridBase<VectorType>> coarse;

  /**
   * Object for grid transfer.
   */
  const MGTransferMF<VectorType> & transfer;

  /**
   * The smoothing object.
   */
  SmartPointer<const MGLevelObject<std::shared_ptr<SmootherType>>> smoother;

  const unsigned int n_cycles;

  DynamicConvergenceTable table;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ */
