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
  PreconditionerBase() : update_needed(true)
  {
  }

  virtual ~PreconditionerBase()
  {
  }

  virtual void
  setup(unsigned int const cell) = 0;

  virtual void
  update() = 0;

  bool
  needs_update()
  {
    return update_needed;
  }

  virtual void
  vmult(Number * dst, Number const * src) const = 0;

protected:
  bool update_needed;
};

/**
 * This class implements an identity preconditioner for iterative solvers for elementwise problems.
 */
template<typename Number>
class PreconditionerIdentity : public PreconditionerBase<Number>
{
public:
  PreconditionerIdentity(unsigned int const size) : M(size)
  {
    this->update_needed = false;
  }

  virtual ~PreconditionerIdentity()
  {
  }

  void
  setup(unsigned int const /* cell */) final
  {
    // nothing to do
  }

  void
  update() final
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

/**
 * This class implements a Jacobi preconditioner for iterative solvers for elementwise problems.
 */
template<int dim, int n_components, typename Number, typename Operator>
class JacobiPreconditioner : public Elementwise::PreconditionerBase<dealii::VectorizedArray<Number>>
{
  typedef CellIntegrator<dim, n_components, Number> Integrator;

public:
  JacobiPreconditioner(dealii::MatrixFree<dim, Number> const & matrix_free,
                       unsigned int const                      dof_index,
                       unsigned int const                      quad_index,
                       Operator const &                        underlying_operator_in,
                       bool const                              initialize)
    : underlying_operator(underlying_operator_in)
  {
    integrator = std::make_shared<Integrator>(matrix_free, dof_index, quad_index);

    underlying_operator.initialize_dof_vector(global_inverse_diagonal);

    if(initialize)
    {
      this->update();
    }
  }

  void
  setup(unsigned int cell) final
  {
    integrator->reinit(cell);
    integrator->read_dof_values(global_inverse_diagonal, 0);
  }

  void
  update() final
  {
    underlying_operator.calculate_inverse_diagonal(global_inverse_diagonal);

    this->update_needed = false;
  }

  /**
   * The pointers dst, src may point to the same data.
   */
  void
  vmult(dealii::VectorizedArray<Number> *       dst,
        dealii::VectorizedArray<Number> const * src) const final
  {
    for(unsigned int i = 0; i < integrator->dofs_per_cell; ++i)
    {
      dst[i] = integrator->begin_dof_values()[i] * src[i];
    }
  }

private:
  std::shared_ptr<Integrator> integrator;

  Operator const & underlying_operator;

  dealii::LinearAlgebra::distributed::Vector<Number> global_inverse_diagonal;
};

/**
 * This class implements an elementwise inverse mass preconditioner. Currently, this class can only
 * be used if the inverse mass can be realized as a matrix-free operator evaluation available via
 * utility functions in deal.II.
 */
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

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();

    // The inverse mass preconditioner is only available for discontinuous Galerkin discretizations.
    AssertThrow(
      fe.conforms(dealii::FiniteElementData<dim>::L2),
      dealii::ExcMessage(
        "The elementwise inverse mass preconditioner is only implemented for DG (L2-conforming) elements."));

    // Currently, the inverse mass realized as matrix-free operator evaluation is only available
    // in deal.II for tensor-product elements.
    AssertThrow(
      fe.base_element(0).dofs_per_cell == dealii::Utilities::pow(fe.degree + 1, dim),
      dealii::ExcMessage(
        "The elementwise inverse mass preconditioner is only implemented for tensor-product DG elements."));

    // Currently, the inverse mass realized as matrix-free operator evaluation is only available
    // in deal.II if n_q_points_1d = n_nodes_1d.
    AssertThrow(
      matrix_free.get_shape_info(0, quad_index).data[0].n_q_points_1d == fe.degree + 1,
      dealii::ExcMessage(
        "The elementwise inverse mass preconditioner is only available if n_q_points_1d = n_nodes_1d."));

    this->update_needed = false;
  }

  void
  setup(unsigned int const cell) final
  {
    integrator->reinit(cell);
  }

  void
  update() final
  {
    // no updates needed as long as the MatrixFree/Integrator object is up-to-date (which is not the
    // responsibility of the present class).
  }

  /**
   * The pointers dst, src may point to the same data.
   */
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
