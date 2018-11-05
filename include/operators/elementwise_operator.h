/*
 * elementwise_operator.h
 *
 *  Created on: Oct 31, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_ELEMENTWISE_OPERATOR_H_
#define INCLUDE_OPERATORS_ELEMENTWISE_OPERATOR_H_

#include "../solvers_and_preconditioners/solvers/elementwise_krylov_solvers.h"

namespace Elementwise
{
template<int dim, typename Number, typename Operator>
class OperatorBase
{
public:
  OperatorBase(Operator const & operator_in) : op(operator_in), current_cell(1), problem_size(1)
  {
  }

  MatrixFree<dim, Number> const &
  get_data() const
  {
    return op.get_data();
  }

  unsigned int
  get_dof_index() const
  {
    return op.get_dof_index();
  }

  unsigned int
  get_quad_index() const
  {
    return op.get_quad_index();
  }

  void
  setup(unsigned int const cell, unsigned int const size)
  {
    current_cell = cell;

    problem_size = size;
  }

  unsigned int
  get_problem_size() const
  {
    return problem_size;
  }

  void
  vmult(VectorizedArray<Number> * dst, VectorizedArray<Number> * src) const
  {
    // set dst vector to zero
    Elementwise::vector_init(dst, problem_size);

    // evaluate block diagonal
    op.apply_add_block_diagonal_elementwise(current_cell, dst, src, problem_size);
  }

private:
  Operator const & op;

  unsigned int current_cell;

  unsigned int problem_size;
};

} // namespace Elementwise


#endif /* INCLUDE_OPERATORS_ELEMENTWISE_OPERATOR_H_ */
