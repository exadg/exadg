/*
 * multigrid_algorithm.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

#include <deal.II/base/function_lib.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

using namespace dealii;

/*
 * Activate timings if desired.
 */

//#define ENABLE_TIMING true

#ifndef ENABLE_TIMING
#  define ENABLE_TIMING false
#endif

using namespace dealii;
/*
 * Re-implementation of multigrid preconditioner (V-cycle) in order to have more direct control over
 * its individual components and avoid inner products and other expensive stuff.
 */
template<typename VectorType,
         typename MatrixType,
         typename TransferType,
         typename PreconditionerType>
class MultigridPreconditioner
{
public:
  MultigridPreconditioner(const MGLevelObject<std::shared_ptr<MatrixType>> &         matrix,
                          const MGCoarseGridBase<VectorType> &                       coarse,
                          const MGTransferBase<VectorType> &                         transfer,
                          const MGLevelObject<std::shared_ptr<PreconditionerType>> & smooth,
                          const unsigned int                                         n_cycles = 1)
    : minlevel(matrix.min_level()),
      maxlevel(matrix.max_level()),
      defect(minlevel, maxlevel),
      solution(minlevel, maxlevel),
      t(minlevel, maxlevel),
      defect2(minlevel, maxlevel),
      matrix(&matrix, typeid(*this).name()),
      coarse(&coarse, typeid(*this).name()),
      transfer(transfer),
      smooth(&smooth, typeid(*this).name()),
      n_cycles(n_cycles)
#if ENABLE_TIMING
      ,
      timer(minlevel, maxlevel),
      wall_time(minlevel, maxlevel)
#endif
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
      wall_time[level] = 0.0;
#endif
  }

  virtual ~MultigridPreconditioner()
  {
#if ENABLE_TIMING
    for(unsigned int level = minlevel; level <= maxlevel; ++level)
      printf(" >>> %d %12.9f\n", level, wall_time[level]);
#endif
  }


  template<class OtherVectorType>
  void
  vmult(OtherVectorType & dst, const OtherVectorType & src) const
  {
#if ENABLE_TIMING
    timer[maxlevel].restart();
#endif

    for(unsigned int i = minlevel; i <= maxlevel; i++)
      defect[i] = 0.0;
    defect[maxlevel].copy_locally_owned_data_from(src);

#if ENABLE_TIMING
    wall_time[maxlevel] += timer[maxlevel].wall_time();
#endif

    v_cycle(maxlevel);

#if ENABLE_TIMING
    timer[maxlevel].restart();
#endif

    dst.copy_locally_owned_data_from(solution[maxlevel]);

#if ENABLE_TIMING
    wall_time[maxlevel] += timer[maxlevel].wall_time();
#endif
  }

private:
  /**
   * Implements the V-cycle
   */
  void
  v_cycle(const unsigned int level) const
  {
#if ENABLE_TIMING
    timer[level].restart();
#endif

    // call coarse grid solver
    if(level == minlevel)
    {
      (*coarse)(level, solution[level], defect[level]);

#if ENABLE_TIMING
      wall_time[level] += timer[level].wall_time();
#endif
      return;
    }

    // smoothing
    (*smooth)[level]->vmult(solution[level], defect[level]);
    (*matrix)[level]->vmult_interface_down(t[level], solution[level]);
    t[level].sadd(-1.0, 1.0, defect[level]);

    // transfer to next level
    transfer.restrict_and_add(level, defect[level - 1], t[level]);

    // coarse grid correction
    v_cycle(level - 1);

    // prolongate
    transfer.prolongate(level, t[level], solution[level - 1]);
    solution[level] += t[level];

    // smooth on the negative part of the residual
    defect[level] *= -1.0;
    (*matrix)[level]->vmult_add_interface_up(defect[level], solution[level]);
    (*smooth)[level]->vmult(t[level], defect[level]);
    solution[level] -= t[level];

#if ENABLE_TIMING
    wall_time[level] += timer[level].wall_time();
#endif
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
   * Object for grid tranfer.
   */
  const MGTransferBase<VectorType> & transfer;

  /**
   * The smoothing object.
   */
  SmartPointer<const MGLevelObject<std::shared_ptr<PreconditionerType>>> smooth;

  const unsigned int n_cycles;

#if ENABLE_TIMING
  mutable MGLevelObject<Timer> timer;
  // measures time on the level (including the coarser levels)
  mutable MGLevelObject<double> wall_time;
#endif
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ */
