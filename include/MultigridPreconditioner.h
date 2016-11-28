/*
 * Preconditioner.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_MULTIGRIDPRECONDITIONER_H_
#define INCLUDE_MULTIGRIDPRECONDITIONER_H_


#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/base/function_lib.h>

// Specialized matrix-free implementation that overloads the copy_to_mg
// function for proper initialization of the vectors in matrix-vector
// products.
template <int dim, typename Operator>
class MGTransferMF : public MGTransferMatrixFree<dim, typename Operator::value_type>
{
public:
  MGTransferMF()
    :
    underlying_operator (0)
  {}

  void set_operator(const MGLevelObject<Operator> &operator_in)
  {
    underlying_operator = &operator_in;
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim>                                               &mg_dof,
              MGLevelObject<parallel::distributed::Vector<typename Operator::value_type> > &dst,
              const InVector                                                               &src) const
  {
    AssertThrow(underlying_operator != 0, ExcNotInitialized());

    for (unsigned int level=dst.min_level();level<=dst.max_level(); ++level)
      (*underlying_operator)[level].initialize_dof_vector(dst[level]);

    MGLevelGlobalTransfer<parallel::distributed::Vector<typename Operator::value_type> >::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<Operator> *underlying_operator;
};

// re-implement the multigrid preconditioner in order to have more direct
// control over its individual components and avoid inner products and other
// expensive stuff
template <int dim, typename VectorType, typename MatrixType, typename TransferType, typename PreconditionerType>
class MultigridPreconditioner
{
public:
  MultigridPreconditioner(const DoFHandler<dim>                                           &dof_handler,
                          const MGLevelObject<MatrixType>                                 &matrix,
                          const MGCoarseGridBase<VectorType>                              &coarse,
                          const TransferType                                              &transfer,
                          const MGLevelObject<std_cxx11::shared_ptr<PreconditionerType> > &smooth,
                          const unsigned int                                              n_cycles = 1)
    :
    dof_handler(&dof_handler),
    minlevel(0),
    maxlevel(dof_handler.get_triangulation().n_global_levels()-1),
    defect(minlevel, maxlevel),
    solution(minlevel, maxlevel),
    t(minlevel, maxlevel),
    defect2(minlevel, maxlevel),
    matrix(&matrix, typeid(*this).name()),
    coarse(&coarse, typeid(*this).name()),
    transfer(&transfer, typeid(*this).name()),
    smooth(&smooth, typeid(*this).name()),
    n_cycles (n_cycles)
  {
    AssertThrow(n_cycles == 1, ExcNotImplemented());
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      matrix[level].initialize_dof_vector(solution[level]);
      defect[level] = solution[level];
      t[level] = solution[level];
      if (n_cycles > 1)
        defect2[level] = solution[level];
    }
  }

  template<class OtherVectorType>
  void vmult (OtherVectorType       &dst,
              const OtherVectorType &src) const
  {
    transfer->copy_to_mg(*dof_handler,
                         defect,
                         src);
    v_cycle(maxlevel);
    transfer->copy_from_mg(*dof_handler,
                           dst,
                           solution);
  }


private:
  /**
   * A pointer to the DoFHandler object
   */
  const SmartPointer<const DoFHandler<dim> > dof_handler;

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
  SmartPointer<const MGLevelObject<MatrixType> > matrix;

  /**
   * The matrix for each level.
   */
  SmartPointer<const MGCoarseGridBase<VectorType> > coarse;

  /**
   * Object for grid tranfer.
   */
  SmartPointer<const TransferType> transfer;

  /**
   * The smoothing object.
   */
  SmartPointer<const MGLevelObject<std_cxx11::shared_ptr<PreconditionerType> > > smooth;

  const unsigned int n_cycles;

  /**
   * Implements the V-cycle
   */
  void v_cycle(const unsigned int level) const
  {
    if (level==minlevel)
    {
      (*coarse)(level, solution[level], defect[level]);
      return;
    }

    (*smooth)[level]->vmult(solution[level], defect[level]);
    (*matrix)[level].vmult_interface_down(t[level], solution[level]);
    t[level].sadd(-1.0, 1.0, defect[level]);

    // transfer to next level
    transfer->restrict_and_add(level, defect[level-1], t[level]);

    v_cycle(level-1);

    transfer->prolongate(level, t[level], solution[level-1]);
    solution[level] += t[level];
    // smooth on the negative part of the residual
    defect[level] *= -1.0;
    (*matrix)[level].vmult_add_interface_up(defect[level], solution[level]);
    (*smooth)[level]->vmult(t[level], defect [level]);
    solution[level] -= t[level];
  }
};


#endif /* INCLUDE_MULTIGRIDPRECONDITIONER_H_ */
