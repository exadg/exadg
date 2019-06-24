/*
 * inverse_mass_matrix_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_

#include "../preconditioner/preconditioner_base.h"
#include "operators/inverse_mass_matrix.h"

template<int dim, int n_components, typename Number>
class InverseMassMatrixPreconditioner : public PreconditionerBase<Number>
{
public:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

  InverseMassMatrixPreconditioner(MatrixFree<dim, Number> const & matrix_free,
                                  unsigned int const              degree,
                                  unsigned int const              dof_index,
                                  unsigned int const              quad_index)
  {
    inverse_mass_matrix_operator.initialize(matrix_free, degree, dof_index, quad_index);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    inverse_mass_matrix_operator.apply(dst, src);
  }

  void
  update()
  {
    // do nothing
  }

private:
  InverseMassMatrixOperator<dim, n_components, Number> inverse_mass_matrix_operator;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_ */
