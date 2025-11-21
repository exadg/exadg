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

// ExaDG
#include <exadg/operators/structure_calculators.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

namespace ExaDG
{
template<int dim, typename Number>
DisplacementJacobianCalculator<dim, Number>::DisplacementJacobianCalculator()
  : matrix_free(nullptr), dof_index_vector(0), dof_index_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_vector_in,
  unsigned int const                      dof_index_scalar_in,
  unsigned int const                      quad_index_in)
{
  matrix_free      = &matrix_free_in;
  dof_index_vector = dof_index_vector_in;
  dof_index_scalar = dof_index_scalar_in;
  quad_index       = quad_index_in;
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::compute_projection_rhs(
  VectorType &       dst_scalar_valued,
  VectorType const & src_vector_valued) const
{
  dst_scalar_valued = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst_scalar_valued, src_vector_valued);
}

template<int dim, typename Number>
void
DisplacementJacobianCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst_scalar_valued,
  VectorType const &                            src_vector_valued,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    // Do not enforce constraints on the `src` vector, as constraints are already applied and
    // `dealii::MatrixFree` object stores constraints relevant in linear systemes, not necessarily
    // constraints suitable for a DoF vector corresponding to the solution (e.g., in Newton's
    // method).
    integrator_vector.read_dof_values_plain(src_vector_valued);
    integrator_vector.evaluate(dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
    {
      tensor const gradient_displacement = integrator_vector.get_gradient(q);
      tensor const F                     = Structure::get_F(gradient_displacement);
      scalar const Jacobian              = determinant(F);

      integrator_scalar.submit_value(Jacobian, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst_scalar_valued);
  }
}

template class DisplacementJacobianCalculator<2, float>;
template class DisplacementJacobianCalculator<2, double>;

template class DisplacementJacobianCalculator<3, float>;
template class DisplacementJacobianCalculator<3, double>;

} // namespace ExaDG
