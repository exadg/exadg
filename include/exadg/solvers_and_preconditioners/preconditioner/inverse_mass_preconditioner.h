/*
 * inverse_mass_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_

#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioner/preconditioner_base.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, int n_components, typename Number>
class InverseMassPreconditioner : public PreconditionerBase<Number>
{
public:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

  InverseMassPreconditioner(MatrixFree<dim, Number> const & matrix_free,
                            unsigned int const              dof_index,
                            unsigned int const              quad_index)
  {
    inverse_mass_operator.initialize(matrix_free, dof_index, quad_index);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    inverse_mass_operator.apply(dst, src);
  }

  void
  update()
  {
    // do nothing
  }

private:
  InverseMassOperator<dim, n_components, Number> inverse_mass_operator;
};
} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_ */
