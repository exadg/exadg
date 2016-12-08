/*
 * JacobiPreconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_JACOBIPRECONDITIONER_H_
#define INCLUDE_JACOBIPRECONDITIONER_H_

#include <deal.II/lac/parallel_vector.h>

#include "PreconditionerBase.h"
#include "MatrixOperatorBase.h"

template<typename value_type, typename UnderlyingOperator>
class JacobiPreconditioner : public PreconditionerBase<value_type>
{
public:
  JacobiPreconditioner(UnderlyingOperator const &underlying_operator)
  {
    underlying_operator.initialize_dof_vector(inverse_diagonal);

    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    if (!PointerComparison::equal(&dst, &src))
      dst = src;
    dst.scale(inverse_diagonal);
  }

  void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const *underlying_operator = dynamic_cast<UnderlyingOperator const *>(matrix_operator);
    if(underlying_operator)
      underlying_operator->calculate_inverse_diagonal(inverse_diagonal);
    else
      AssertThrow(false,ExcMessage("Jacobi preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));
  }

  unsigned int get_size_of_diagonal()
  {
    return inverse_diagonal.size();
  }

private:
  parallel::distributed::Vector<value_type> inverse_diagonal;
};


#endif /* INCLUDE_JACOBIPRECONDITIONER_H_ */
