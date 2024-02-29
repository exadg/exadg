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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_BLOCK_PRECONDITIONER_PROJECTION_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_BLOCK_PRECONDITIONER_PROJECTION_H_

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/projection_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Block preconditioner for projection operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename Number>
class BlockPreconditionerProjection : public PreconditionerBase<Number>
{
protected:
  typedef ProjectionOperator<dim, Number>                    PDEOperator;
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

private:
  dealii::FullMatrix<Number>                                            inverse_mass_matrix;
  dealii::AlignedVector<scalar>                                         transformation_value;
  dealii::AlignedVector<scalar>                                         transformation_deriv;
  dealii::AlignedVector<scalar>                                         transformation_eigenvalues;
  dealii::AlignedVector<std::pair<scalar, std::array<scalar, 2 * dim>>> penalty_parameters;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  PDEOperator const *                     pde_operator;

public:
  /*
   * Constructor.
   */
  BlockPreconditionerProjection(dealii::MatrixFree<dim, Number> const & matrix_free,
                                PDEOperator const &                     pde_operator);

  /*
   * Update of preconditioner.
   */
  void
  update() override;

  /*
   * This function applies the multigrid preconditioner dst = P^{-1} src.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const override;

  /*
   * This function applies the multigrid preconditioner dst = P^{-1} src and
   * embedds the work schedule before/after the application (if supported by
   * the preconditioner).
   */
  virtual void
  vmult(
    VectorType &                                                        dst,
    VectorType const &                                                  src,
    const std::function<void(const unsigned int, const unsigned int)> & before_loop,
    const std::function<void(const unsigned int, const unsigned int)> & after_loop) const override;

private:
  template<int degree>
  void
  do_vmult(VectorType &                                                        dst,
           VectorType const &                                                  src,
           const std::function<void(const unsigned int, const unsigned int)> & before_loop,
           const std::function<void(const unsigned int, const unsigned int)> & after_loop) const;
};
} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_BLOCK_PRECONDITIONER_PROJECTION_H_ \
        */
