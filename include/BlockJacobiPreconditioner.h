/*
 * BlockJacobiPreconditioner.h
 *
 *  Created on: Nov 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_BLOCKJACOBIPRECONDITIONER_H_
#define INCLUDE_BLOCKJACOBIPRECONDITIONER_H_

#include "PreconditionerBase.h"
#include "MatrixOperatorBase.h"

template<int dim, typename value_type, typename UnderlyingOperator>
class BlockJacobiPreconditioner : public PreconditionerBase<value_type>
{
public:
  BlockJacobiPreconditioner(UnderlyingOperator const &underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {}

  void update(MatrixOperatorBase const * matrix_operator){}

  void vmult (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    underlying_operator.apply_block_jacobi(dst,src);
  }

private:
  UnderlyingOperator const &underlying_operator;
};


#endif /* INCLUDE_BLOCKJACOBIPRECONDITIONER_H_ */
