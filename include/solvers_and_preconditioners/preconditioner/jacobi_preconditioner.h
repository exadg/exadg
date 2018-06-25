/*
 * JacobiPreconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBIPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBIPRECONDITIONER_H_

#include <deal.II/lac/parallel_vector.h>

#include "./preconditioner_base.h"

#include "../../operators/matrix_operator_base.h"
#include "../../operators/matrix_operator_base_new.h"

template<typename Operator>
class JacobiPreconditioner : public PreconditionerBase<typename Operator::value_type>
{
public:
    
    typedef typename Operator::value_type value_type;
    
  JacobiPreconditioner(Operator const &underlying_operator)
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
    Operator const *underlying_operator = dynamic_cast<Operator const *>(matrix_operator);
    if(underlying_operator)
      underlying_operator->calculate_inverse_diagonal(inverse_diagonal);
    else
      AssertThrow(false,ExcMessage("Jacobi preconditioner: MatrixOperatorBaseNew<dim, value_type> and MatrixOperator are not compatible!"));
  }

  unsigned int get_size_of_diagonal()
  {
    return inverse_diagonal.size();
  }

private:
  parallel::distributed::Vector<value_type> inverse_diagonal;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBIPRECONDITIONER_H_ */
