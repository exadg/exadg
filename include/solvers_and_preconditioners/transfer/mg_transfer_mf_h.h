/*
 * mg_transfer_mf_h.h
 *
 *  Created on: Nov 29, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_TRANSFER_MG_TRANSFER_MF_H_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_TRANSFER_MG_TRANSFER_MF_H_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "../../operators/multigrid_operator_base.h"

using namespace dealii;

// Specialized matrix-free implementation that overloads the copy_to_mg
// function for proper initialization of the vectors in matrix-vector products.
template<int dim, typename Number>
class MGTransferMF : public MGTransferMatrixFree<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MGTransferMF(std::map<unsigned int, unsigned int> level_to_triangulation_level_map)
    : underlying_operator(0), level_to_triangulation_level_map(level_to_triangulation_level_map)
  {
  }

  virtual ~MGTransferMF()
  {
  }

  void
  set_operator(
    const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> & operator_in)
  {
    underlying_operator = &operator_in;
  }

  virtual void
  prolongate(unsigned int const to_level, VectorType & dst, VectorType const & src) const
  {
    MGTransferMatrixFree<dim, Number>::prolongate(level_to_triangulation_level_map[to_level],
                                                  dst,
                                                  src);
  }

  virtual void
  restrict_and_add(unsigned int const from_level, VectorType & dst, VectorType const & src) const
  {
    MGTransferMatrixFree<dim, Number>::restrict_and_add(
      level_to_triangulation_level_map[from_level], dst, src);
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template<class InVector, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> & mg_dof,
             MGLevelObject<VectorType> &       dst,
             const InVector &                  src) const
  {
    AssertThrow(underlying_operator != 0, ExcNotInitialized());

    for(unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
      (*underlying_operator)[level]->initialize_dof_vector(dst[level]);

    MGLevelGlobalTransfer<VectorType>::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> * underlying_operator;

  // this map converts the multigrid level as used in the V-cycle to an actual
  // level in the triangulation (this is necessary since both numbers might not
  // equal e.g. in the case of hp-MG multiple (p-)levels
  // are on the zeroth triangulation level)
  mutable std::map<unsigned int, unsigned int> level_to_triangulation_level_map;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_TRANSFER_MG_TRANSFER_MF_H_H_ */
