/*
 * inverse_mass_matrix_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "../../operators/linear_operator_base.h"
#include "../preconditioner/preconditioner_base.h"
#include "operators/inverse_mass_matrix.h"

template<int dim, int degree, typename Number, int n_components>
class InverseMassMatrixPreconditioner : public PreconditionerBase<Number>
{
public:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

  InverseMassMatrixPreconditioner(MatrixFree<dim, Number> const & mf_data,
                                  unsigned int const              dof_index,
                                  unsigned int const              quad_index)
  {
    inverse_mass_matrix_operator.initialize(mf_data, dof_index, quad_index);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    inverse_mass_matrix_operator.apply(dst, src);
  }

  void
  update(LinearOperatorBase const * /*linear_operator*/)
  {
    // do nothing
  }

private:
  InverseMassMatrixOperator<dim, degree, Number, n_components> inverse_mass_matrix_operator;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_ */
