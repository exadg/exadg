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

#include "transfer/mg_transfer_mf.h"

using namespace dealii;

/*
 * Activate timings if desired.
 */
#define ENABLE_TIMING false

#ifndef ENABLE_TIMING
#  define ENABLE_TIMING false
#endif

using namespace dealii;

class MultigridTimings
{
public:
  MultigridTimings() : min_level(0), max_level(0)
  {
  }

  void
  init(unsigned int const minlevel, unsigned int const maxlevel) const
  {
    vec.resize(maxlevel - minlevel + 1);
    min_level = minlevel;
    max_level = maxlevel;
  }

  void
  add(unsigned int const level, std::string const label, double const value) const
  {
    auto & map = vec[level - min_level];
    auto   it  = map.find(label);
    if(it != map.end())
      it->second += value;
    else
      map[label] = value;
  }

  void
  set(unsigned int const level, std::string const label, double const value) const
  {
    auto & map = vec[level - min_level];
    auto   it  = map.find(label);
    if(it != map.end())
      it->second = value;
    else
      map[label] = value;
  }

  void
  print() const
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank)
      return;

    std::cout << "Wall times [sec] ('overall' includes current level and all coarser levels):"
              << std::endl
              << std::endl;

    unsigned int level = max_level;
    for(std::vector<std::map<std::string, double>>::reverse_iterator it = vec.rbegin();
        it != vec.rend();
        ++it)
    {
      std::cout << std::setw(6) << "level";

      std::map<std::string, double> & map = *it;
      for(auto iter_map : map)
        std::cout << std::setw(12) << iter_map.first;

      std::cout << std::endl;

      std::cout << std::setw(6) << level;
      --level;

      for(auto iter_map : map)
        std::cout << std::setw(12) << std::scientific << std::setprecision(2) << iter_map.second;

      std::cout << std::endl << std::endl;
    }
  }

private:
  mutable unsigned int                               min_level;
  mutable unsigned int                               max_level;
  mutable std::vector<std::map<std::string, double>> vec;
};

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
    timings.init(minlevel, maxlevel);

    for(unsigned int level = minlevel; level <= maxlevel; ++level)
      timings.set(level, "dofs", defect[level].size());
#endif
  }

  virtual ~MultigridPreconditioner()
  {
#if ENABLE_TIMING
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      timings.print();
#endif
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
  /**
   * Implements the V-cycle
   */
  void
  v_cycle(unsigned int const level, bool const multigrid_is_a_solver) const
  {
#if ENABLE_TIMING
    Timer timer_local;
    Timer timer_global;
    timer_global.restart();
#endif

    // call coarse grid solver
    if(level == minlevel)
    {
      (*coarse)(level, solution[level], defect[level]);

#if ENABLE_TIMING
      timings.add(level, "overall", timer_global.wall_time());
#endif
    }
    else
    {
#if ENABLE_TIMING
      timer_local.restart();
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

#if ENABLE_TIMING
      timings.add(level, "presmooth", timer_local.wall_time());
#endif

#if ENABLE_TIMING
      timer_local.restart();
#endif

      // restriction
      (*matrix)[level]->vmult_interface_down(t[level], solution[level]);
      t[level].sadd(-1.0, 1.0, defect[level]);
      transfer.restrict_and_add(level, defect[level - 1], t[level]);

#if ENABLE_TIMING
      timings.add(level, "restrict", timer_local.wall_time());
#endif

      // coarse grid correction
      v_cycle(level - 1, false);

#if ENABLE_TIMING
      timer_local.restart();
#endif

      // prolongation
      transfer.prolongate(level, t[level], solution[level - 1]);
      solution[level] += t[level];

#if ENABLE_TIMING
      timings.add(level, "prolong", timer_local.wall_time());
#endif

#if ENABLE_TIMING
      timer_local.restart();
#endif

      // post-smoothing
      (*smoother)[level]->step(solution[level], defect[level]);

#if ENABLE_TIMING
      timings.add(level, "postsmooth", timer_local.wall_time());
      timings.add(level, "overall", timer_global.wall_time());
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

#if ENABLE_TIMING
  MultigridTimings timings;
#endif
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ */
