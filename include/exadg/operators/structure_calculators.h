/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#ifndef EXADG_OPERATORS_STRUCTURE_CALCULATORS_H_
#define EXADG_OPERATORS_STRUCTURE_CALCULATORS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
/*
 * Calculator for the Jacobian of the displacement field u(X) in material coordinates X defined as
 *
 * J = det(F) with F = I + Grad(u),
 *
 * where F is the deformation gradient tensor and Grad(u) is the gradient with respect to the
 * material coordinates X of the displacement field.
 *
 */
template<int dim, typename Number>
class DisplacementJacobianCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DisplacementJacobianCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  DisplacementJacobianCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_vector_in,
             unsigned int const                      dof_index_scalar_in,
             unsigned int const                      quad_index_in);

  /*
   * Compute the right-hand side of an L2 projection of the Jacobian of the displacement field.
   */
  void
  compute_projection_rhs(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free,
            VectorType &                                  dst_scalar_valued,
            VectorType const &                            src_vector_valued,
            std::pair<unsigned int, unsigned int> const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_STRUCTURE_CALCULATORS_H_ */
