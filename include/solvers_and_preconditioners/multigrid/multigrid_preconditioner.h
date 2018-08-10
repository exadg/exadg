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

//#define ENABLE_TIMING true

#ifndef ENABLE_TIMING
    #define ENABLE_TIMING false
#endif

// Specialized matrix-free implementation that overloads the copy_to_mg
// function for proper initialization of the vectors in matrix-vector
// products.
using namespace dealii;

template <int dim, typename Type>
class MGTransferMF : public MGTransferMatrixFree<dim, Type>
{
public:
  MGTransferMF(std::map<unsigned int, unsigned int> m)
    :
    underlying_operator (0),m(m)
  {}

  void set_operator(const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Type>>> &operator_in)
  {
    underlying_operator = &operator_in;
  }
  
  virtual void prolongate (const unsigned int                           to_level,
                           LinearAlgebra::distributed::Vector<Type>       &dst,
                           const LinearAlgebra::distributed::Vector<Type> &src) const{
      unsigned int temp = to_level;
      MGTransferMatrixFree<dim, Type>::prolongate(m[temp], dst, src);
  }
  
  virtual void restrict_and_add (const unsigned int from_level,
                                 LinearAlgebra::distributed::Vector<Type>       &dst,
                                 const LinearAlgebra::distributed::Vector<Type> &src) const{
      unsigned int temp = from_level;
      MGTransferMatrixFree<dim, Type>::restrict_and_add(m[temp], dst, src);
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim>                                               &mg_dof,
              MGLevelObject<parallel::distributed::Vector<Type> > &dst,
              const InVector                                                               &src) const
  {
    AssertThrow(underlying_operator != 0, ExcNotInitialized());

    for (unsigned int level=dst.min_level();level<=dst.max_level(); ++level)
      (*underlying_operator)[level]->initialize_dof_vector(dst[level]);

    MGLevelGlobalTransfer<parallel::distributed::Vector<Type> >::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Type>>> *underlying_operator;
  mutable std::map<unsigned int, unsigned int> m;
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
  , time (minlevel, maxlevel)
  , ctime(minlevel, maxlevel)
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
        ctime[level] = 0.0;
#endif
  }
    
    virtual ~MultigridPreconditioner() {
        
#if ENABLE_TIMING
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
        printf(" >>> %d %12.9f\n", level, ctime[level]); 
#endif
    
    }


  template<class OtherVectorType>
  void vmult (OtherVectorType       &dst,
              const OtherVectorType &src) const
  {
      //std::cout << "MultigridPreconditioner:v:mult::1" << std::endl;
#if ENABLE_TIMING
      time[maxlevel].restart();
#endif
    for(unsigned int i = minlevel; i <= maxlevel; i++)
      defect[i] = 0.0;
    defect[maxlevel].copy_locally_owned_data_from(src);
#if ENABLE_TIMING
    ctime[maxlevel] += time[maxlevel].wall_time();
#endif
      //std::cout << "MultigridPreconditioner:v:mult::2" << std::endl;
    v_cycle(maxlevel);
      //std::cout << "MultigridPreconditioner:v:mult::3" << std::endl;
#if ENABLE_TIMING
    time[maxlevel].restart();
#endif
    dst.copy_locally_owned_data_from(solution[maxlevel]);
#if ENABLE_TIMING
    ctime[maxlevel] += time[maxlevel].wall_time();
#endif
      //std::cout << "MultigridPreconditioner:v:mult::4" << std::endl;
      
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
  mutable MGLevelObject<Timer> time;
  mutable MGLevelObject<double> ctime;
#endif

  /**
   * Implements the V-cycle
   */
  void v_cycle(const unsigned int level) const
  {
#if ENABLE_TIMING
    time[level].restart();
#endif
    if (level==minlevel)
    {
      //std::cout << "MultigridPreconditioner::v_cycle::1" << std::endl;
      (*coarse)(level, solution[level], defect[level]);
      //std::cout << "MultigridPreconditioner::v_cycle::2" << std::endl;
#if ENABLE_TIMING
      ctime[level] += time[level].wall_time();
#endif
      return;
    }

      //std::cout << "MultigridPreconditioner::v_cycle::3" << std::endl;
    (*smooth)[level]->vmult(solution[level], defect[level]);
      //std::cout << "MultigridPreconditioner::v_cycle::4" << std::endl;
    (*matrix)[level]->vmult_interface_down(t[level], solution[level]);
      //std::cout << "MultigridPreconditioner::v_cycle::5" << std::endl;
      //std::cout << "MultigridPreconditioner::v_cycle::5" << std::endl;
    t[level].sadd(-1.0, 1.0, defect[level]);
      //std::cout << "MultigridPreconditioner::v_cycle::6" << std::endl;

    // transfer to next level
    (*transfer)[level]->restrict_and_add(level, defect[level-1], t[level]);
      //std::cout << "MultigridPreconditioner::v_cycle::7" << std::endl;

    v_cycle(level-1);
      //std::cout << "MultigridPreconditioner::v_cycle::8" << std::endl;

    (*transfer)[level]->prolongate(level, t[level], solution[level-1]);
      //std::cout << "MultigridPreconditioner::v_cycle::9" << std::endl;
    solution[level] += t[level];
      //std::cout << "MultigridPreconditioner::v_cycle::10" << std::endl;
    // smooth on the negative part of the residual
    defect[level] *= -1.0;
      //std::cout << "MultigridPreconditioner::v_cycle::11" << std::endl;
    (*matrix)[level]->vmult_add_interface_up(defect[level], solution[level]);
      //std::cout << "MultigridPreconditioner::v_cycle::12" << std::endl;
    (*smooth)[level]->vmult(t[level], defect [level]);
      //std::cout << "MultigridPreconditioner::v_cycle::13" << std::endl;
    solution[level] -= t[level];
      //std::cout << "MultigridPreconditioner::v_cycle::14" << std::endl;
#if ENABLE_TIMING
    ctime[level] += time[level].wall_time();
#endif
  }
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ */
