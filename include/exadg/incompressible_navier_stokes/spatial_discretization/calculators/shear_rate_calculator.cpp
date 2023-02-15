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

#include <exadg/incompressible_navier_stokes/spatial_discretization/calculators/shear_rate_calculator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
ShearRateCalculator<dim, Number>::ShearRateCalculator()
  : matrix_free(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
ShearRateCalculator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                             unsigned int const                      dof_index_u_in,
                                             unsigned int const dof_index_u_scalar_in,
                                             unsigned int const quad_index_in)
{
  matrix_free        = &matrix_free_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, typename Number>
void
ShearRateCalculator<dim, Number>::compute_shear_rate(VectorType & dst, VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
ShearRateCalculator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                            VectorType &                            dst,
                                            VectorType const &                      src,
                                            Range const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_u, quad_index);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_scalar.n_q_points; q++)
    {
      dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> sym_grad_u =
        integrator_vector.get_symmetric_gradient(q);
      scalar shear_rate = std::sqrt(0.5 * scalar_product(sym_grad_u, sym_grad_u));

      integrator_scalar.submit_value(shear_rate, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template class ShearRateCalculator<2, float>;
template class ShearRateCalculator<2, double>;

template class ShearRateCalculator<3, float>;
template class ShearRateCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
