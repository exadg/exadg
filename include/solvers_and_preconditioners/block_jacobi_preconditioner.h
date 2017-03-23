/*
 * BlockJacobiPreconditioner.h
 *
 *  Created on: Nov 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCKJACOBIPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCKJACOBIPRECONDITIONER_H_

#include <deal.II/lac/parallel_vector.h>

#include "operators/matrix_operator_base.h"
#include "preconditioner_base.h"

template<int dim, typename value_type, typename UnderlyingOperator>
class BlockJacobiPreconditioner : public PreconditionerBase<value_type>
{
public:
  BlockJacobiPreconditioner(UnderlyingOperator const &underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {
    // initialize block Jacobi
    underlying_operator.update_block_jacobi();
  }

  /*
   *  This function updates the block Jacobi preconditioner.
   *  Make sure that the underlying operator has been updated
   *  when calling this function.
   */
  void update(MatrixOperatorBase const * /*matrix_operator*/)
  {
    underlying_operator.update_block_jacobi();
  }

  /*
   *  This function applies the block Jacobi preconditioner.
   *  Make sure that the block Jacobi preconditioner has been
   *  updated when calling this function.
   */
  void vmult (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    underlying_operator.apply_block_jacobi(dst,src);
  }

private:
  UnderlyingOperator const &underlying_operator;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCKJACOBIPRECONDITIONER_H_ */
