/*
 * block_jacobi_preconditioner.h
 *
 *  Created on: Nov 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCKJACOBIPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCKJACOBIPRECONDITIONER_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "./preconditioner_base.h"

#include "../../operators/matrix_operator_base.h"

template<typename Operator>
class BlockJacobiPreconditioner : public PreconditionerBase<typename Operator::value_type>
{
public:
  typedef typename PreconditionerBase<typename Operator::value_type>::VectorType VectorType;

  BlockJacobiPreconditioner(Operator const & underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {
    // initialize block Jacobi
    underlying_operator.update_block_diagonal_preconditioner();
  }

  /*
   *  This function updates the block Jacobi preconditioner.
   *  Make sure that the underlying operator has been updated
   *  when calling this function.
   */
  void
  update(MatrixOperatorBase const * /*matrix_operator*/)
  {
    underlying_operator.update_block_diagonal_preconditioner();
  }

  /*
   *  This function applies the block Jacobi preconditioner.
   *  Make sure that the block Jacobi preconditioner has been
   *  updated when calling this function.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    underlying_operator.apply_inverse_block_diagonal(dst, src);
  }

private:
  Operator const & underlying_operator;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCKJACOBIPRECONDITIONER_H_ */
