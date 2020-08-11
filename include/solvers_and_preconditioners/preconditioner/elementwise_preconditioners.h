/*
 * elementwise_preconditioners.h
 *
 *  Created on: Oct 24, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_

#include <deal.II/matrix_free/operators.h>
#include "../../matrix_free/integrators.h"

#include "../solvers/elementwise_krylov_solvers.h"

namespace Elementwise
{
/*
 * Preconditioners
 */
template<typename Number>
class PreconditionerBase
{
public:
  PreconditionerBase()
  {
  }

  virtual ~PreconditionerBase()
  {
  }

  virtual void
  setup(unsigned int const cell) = 0;

  virtual void
  vmult(Number * dst, Number const * src) const = 0;

private:
};

template<typename Number>
class PreconditionerIdentity : public PreconditionerBase<Number>
{
public:
  PreconditionerIdentity(unsigned int const size) : M(size)
  {
  }

  virtual ~PreconditionerIdentity()
  {
  }

  virtual void
  setup(unsigned int const /* cell */)
  {
    // nothing to do
  }

  virtual void
  vmult(Number * dst, Number const * src) const
  {
    Number one;
    one = 1.0;
    equ(dst, one, src, M);
  }

private:
  unsigned int const M;
};

template<int dim, int n_components, typename Number>
class InverseMassMatrixPreconditioner
  : public Elementwise::PreconditionerBase<VectorizedArray<Number>>
{
public:
  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  InverseMassMatrixPreconditioner(MatrixFree<dim, Number> const & matrix_free,
                                  unsigned int const              dof_index,
                                  unsigned int const              quad_index)
  {
    fe_eval.reset(new Integrator(matrix_free, dof_index, quad_index));
    inverse.reset(new CellwiseInverseMass(*fe_eval));

    coefficients.resize(fe_eval->n_q_points);
  }

  void
  setup(const unsigned int cell)
  {
    fe_eval->reinit(cell);
    inverse->fill_inverse_JxW_values(coefficients);
  }

  void
  vmult(VectorizedArray<Number> * dst, VectorizedArray<Number> const * src) const
  {
    inverse->apply(coefficients, n_components, src, dst);
  }

private:
  std::shared_ptr<Integrator> fe_eval;

  AlignedVector<VectorizedArray<Number>> coefficients;

  std::shared_ptr<CellwiseInverseMass> inverse;
};

} // namespace Elementwise



#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_ */
