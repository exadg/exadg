/*
 * PreconditionerBase.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_

using namespace dealii;

#include <deal.II/lac/parallel_vector.h>

#include "../../operators/matrix_operator_base.h"

template<typename value_type>
class PreconditionerBase
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  virtual ~PreconditionerBase()
  {
  }

  virtual void
  vmult(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  update(MatrixOperatorBase const * matrix_operator) = 0;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_BASE_H_ */
