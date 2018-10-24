/*
 * elementwise_preconditioners.h
 *
 *  Created on: Oct 24, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include "../solvers/elementwise_krylov_solvers.h"

namespace Elementwise
{
/*
 * Preconditioners
 */
template<typename value_type>
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
  vmult(value_type * dst, value_type const * src) const = 0;

private:
};

template<typename value_type>
class PreconditionerIdentity : public PreconditionerBase<value_type>
{
public:
  PreconditionerIdentity(unsigned int const size) : M(size)
  {
  }

  virtual ~PreconditionerIdentity()
  {
  }

  virtual void
  setup(unsigned int const cell)
  {
    // nothing to do
  }

  virtual void
  vmult(value_type * dst, value_type const * src) const
  {
    value_type one;
    one = 1.0;
    equ(dst, one, src, M);
  }

private:
  unsigned int const M;
};

template<int dim, int number_of_equations, int fe_degree, typename value_type>
class InverseMassMatrixPreconditioner
  : public Elementwise::PreconditionerBase<VectorizedArray<value_type>>
{
public:
  InverseMassMatrixPreconditioner(MatrixFree<dim, value_type> const & data,
                                  unsigned int const                  dof_index,
                                  unsigned int const                  quad_index)
    : fe_eval(
        1,
        FEEvaluation<dim, fe_degree, fe_degree + 1, number_of_equations, value_type>(data,
                                                                                     dof_index,
                                                                                     quad_index)),
      inverse(fe_eval[0])
  {
    coefficients.resize(fe_eval[0].n_q_points);
  }

  void
  setup(const unsigned int cell)
  {
    fe_eval[0].reinit(cell);
    inverse.fill_inverse_JxW_values(coefficients);
  }

  void
  vmult(VectorizedArray<value_type> * dst, VectorizedArray<value_type> const * src) const
  {
    inverse.apply(coefficients, number_of_equations, src, dst);
  }

private:
  mutable AlignedVector<
    FEEvaluation<dim, fe_degree, fe_degree + 1, number_of_equations, value_type>>
    fe_eval;

  AlignedVector<VectorizedArray<value_type>> coefficients;

  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_of_equations, value_type>
    inverse;
};

} // namespace Elementwise



#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_ */
