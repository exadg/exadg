/*
 * elementwise_preconditioners.h
 *
 *  Created on: Oct 24, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_

// deal.II
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/solvers_and_preconditioners/solvers/elementwise_krylov_solvers.h>

namespace ExaDG
{
namespace Elementwise
{
using namespace dealii;

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
class InverseMassPreconditioner : public Elementwise::PreconditionerBase<VectorizedArray<Number>>
{
public:
  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  InverseMassPreconditioner(MatrixFree<dim, Number> const & matrix_free,
                            unsigned int const              dof_index,
                            unsigned int const              quad_index)
  {
    integrator.reset(new Integrator(matrix_free, dof_index, quad_index));
    inverse.reset(new CellwiseInverseMass(*integrator));
  }

  void
  setup(const unsigned int cell)
  {
    integrator->reinit(cell);
  }

  void
  vmult(VectorizedArray<Number> * dst, VectorizedArray<Number> const * src) const
  {
    inverse->apply(src, dst);
  }

private:
  std::shared_ptr<Integrator> integrator;

  std::shared_ptr<CellwiseInverseMass> inverse;
};

} // namespace Elementwise
} // namespace ExaDG


#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONER_ELEMENTWISE_PRECONDITIONERS_H_ \
        */
