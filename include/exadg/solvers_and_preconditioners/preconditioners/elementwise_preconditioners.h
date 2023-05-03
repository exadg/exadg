/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
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

  void
  setup(unsigned int const /* cell */) final
  {
    // nothing to do
  }

  virtual void
  vmult(Number * dst, Number const * src) const final
  {
    Number one;
    one = 1.0;
    equ(dst, one, src, M);
  }

private:
  unsigned int const M;
};

template<int dim, int n_components, typename Number>
class InverseMassPreconditioner
  : public Elementwise::PreconditionerBase<dealii::VectorizedArray<Number>>
{
public:
  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  InverseMassPreconditioner(dealii::MatrixFree<dim, Number> const & matrix_free,
                            unsigned int const                      dof_index,
                            unsigned int const                      quad_index)
  {
    integrator = std::make_shared<Integrator>(matrix_free, dof_index, quad_index);
    inverse    = std::make_shared<CellwiseInverseMass>(*integrator);
  }

  void
  setup(unsigned int const cell) final
  {
    integrator->reinit(cell);
  }

  void
  vmult(dealii::VectorizedArray<Number> *       dst,
        dealii::VectorizedArray<Number> const * src) const final
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
