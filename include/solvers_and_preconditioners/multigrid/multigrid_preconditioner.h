/*
 * Preconditioner.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

#include <deal.II/lac/parallel_vector.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/base/function_lib.h>

#include "../../operators/multigrid_operator_base.h"

//#define ENABLE_TIMING true

#ifndef ENABLE_TIMING
    #define ENABLE_TIMING false
#endif

using namespace dealii;

// Specialized matrix-free implementation that overloads the copy_to_mg
// function for proper initialization of the vectors in matrix-vector
// products.
template <int dim, typename Number>
class MGTransferMF : public MGTransferMatrixFree<dim, Number>
{
public:
  MGTransferMF(std::map<unsigned int, unsigned int> level_to_triangulation_level_map)
    :
    underlying_operator (0),level_to_triangulation_level_map(level_to_triangulation_level_map)
  {}

  void set_operator(const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> &operator_in)
  {
    underlying_operator = &operator_in;
  }
  
  virtual void prolongate (const unsigned int                           to_level,
                           LinearAlgebra::distributed::Vector<Number>       &dst,
                           const LinearAlgebra::distributed::Vector<Number> &src) const{
      MGTransferMatrixFree<dim, Number>::prolongate(level_to_triangulation_level_map[to_level], dst, src);
  }
  
  virtual void restrict_and_add (const unsigned int from_level,
                                 LinearAlgebra::distributed::Vector<Number>       &dst,
                                 const LinearAlgebra::distributed::Vector<Number> &src) const{
      MGTransferMatrixFree<dim, Number>::restrict_and_add(level_to_triangulation_level_map[from_level], dst, src);
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim>                                               &mg_dof,
              MGLevelObject<parallel::distributed::Vector<Number> > &dst,
              const InVector                                                               &src) const
  {
    AssertThrow(underlying_operator != 0, ExcNotInitialized());

    for (unsigned int level=dst.min_level();level<=dst.max_level(); ++level)
      (*underlying_operator)[level]->initialize_dof_vector(dst[level]);

    MGLevelGlobalTransfer<parallel::distributed::Vector<Number> >::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> *underlying_operator;
  
  // this map converts the multigrid level as used in the V-cycle to an actual
  // level in the triangulation (this is necessary since both numbers might not 
  // equal e.g. in the case of hp-MG equal: multiple (p-)levels 
  // are on the zeroth triangulation level)
  mutable std::map<unsigned int, unsigned int> level_to_triangulation_level_map;
};

// re-implement the multigrid preconditioner in order to have more direct
// control over its individual components and avoid inner products and other
// expensive stuff
template <typename VectorType, typename MatrixType, typename TransferType, typename PreconditionerType>
class MultigridPreconditioner
{
public:
  MultigridPreconditioner(const MGLevelObject<std::shared_ptr<MatrixType> >                                 &matrix,
                          const MGCoarseGridBase<VectorType>                              &coarse,
                          const MGLevelObject<std::shared_ptr<TransferType      > > &transfer,
                          const MGLevelObject<std::shared_ptr<PreconditionerType> > &smooth,
                          const unsigned int                                              n_cycles = 1)
    :
    minlevel(matrix.min_level()), maxlevel(matrix.max_level()),
    defect(minlevel, maxlevel),
    solution(minlevel, maxlevel),
    t(minlevel, maxlevel),
    defect2(minlevel, maxlevel),
    matrix(&matrix, typeid(*this).name()),
    coarse(&coarse, typeid(*this).name()),
    transfer(&transfer, typeid(*this).name()),
    smooth(&smooth, typeid(*this).name()),
    n_cycles (n_cycles)
#if ENABLE_TIMING
  , timer (minlevel, maxlevel)
  , wall_time(minlevel, maxlevel)
#endif
  {
    AssertThrow(n_cycles == 1, ExcNotImplemented());
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      matrix[level]->initialize_dof_vector(solution[level]);
      defect[level] = solution[level];
      t[level] = solution[level];
      if (n_cycles > 1)
        defect2[level] = solution[level];
    }
#if ENABLE_TIMING
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
        wall_time[level] = 0.0;
#endif
  }
    
    virtual ~MultigridPreconditioner() {
        
#if ENABLE_TIMING
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
        printf(" >>> %d %12.9f\n", level, wall_time[level]); 
#endif
    
    }


  template<class OtherVectorType>
  void vmult (OtherVectorType       &dst,
              const OtherVectorType &src) const
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
  SmartPointer<const MGLevelObject<std::shared_ptr<MatrixType> > > matrix;

  /**
   * The matrix for each level.
   */
  SmartPointer<const MGCoarseGridBase<VectorType> > coarse;

  /**
   * Object for grid tranfer.
   */
  SmartPointer<const MGLevelObject<std::shared_ptr<TransferType> > > transfer;

  /**
   * The smoothing object.
   */
  SmartPointer<const MGLevelObject<std::shared_ptr<PreconditionerType> > > smooth;

  const unsigned int n_cycles;
  
#if ENABLE_TIMING
  mutable MGLevelObject<Timer> timer;
  mutable MGLevelObject<double> wall_time; // measures time on the level (including the coarser levels)
#endif

  /**
   * Implements the V-cycle
   */
  void v_cycle(const unsigned int level) const
  {
#if ENABLE_TIMING
    timer[level].restart();
#endif
    if (level==minlevel)
    {
      (*coarse)(level, solution[level], defect[level]);
#if ENABLE_TIMING
      wall_time[level] += timer[level].wall_time();
#endif
      return;
    }

    (*smooth)[level]->vmult(solution[level], defect[level]);
    (*matrix)[level]->vmult_interface_down(t[level], solution[level]);
    t[level].sadd(-1.0, 1.0, defect[level]);

    // transfer to next level
    (*transfer)[level]->restrict_and_add(level, defect[level-1], t[level]);

    v_cycle(level-1);

    (*transfer)[level]->prolongate(level, t[level], solution[level-1]);
    solution[level] += t[level];
    // smooth on the negative part of the residual
    defect[level] *= -1.0;
    (*matrix)[level]->vmult_add_interface_up(defect[level], solution[level]);
    (*smooth)[level]->vmult(t[level], defect [level]);
    solution[level] -= t[level];
#if ENABLE_TIMING
    wall_time[level] += timer[level].wall_time();
#endif
  }
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ */
