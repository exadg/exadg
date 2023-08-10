/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/incompressible_neo_hookean.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
IncompressibleNeoHookean<dim, Number>::IncompressibleNeoHookean(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  IncompressibleNeoHookeanData<dim> const &      data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr)
{
  // initialize (potentially variable) shear modulus
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored = dealii::make_vectorized_array<Number>(shear_modulus);

  if(shear_modulus_is_variable)
  {
    // allocate vectors for variable coefficients and initialize with constant values
    shear_modulus_coefficients.initialize(matrix_free, quad_index, false, false);
    shear_modulus_coefficients.set_coefficients(shear_modulus);

    VectorType dummy;
    matrix_free.cell_loop(&IncompressibleNeoHookean<dim, Number>::cell_loop_set_coefficients,
                          this,
                          dummy,
                          dummy);
  }
}

template<int dim, typename Number>
void
IncompressibleNeoHookean<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const &,
  Range const & cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      dealii::VectorizedArray<Number> shear_modulus_vec =
        FunctionEvaluator<0, dim, Number>::value(*(data.shear_modulus_function),
                                                 integrator.quadrature_point(q),
                                                 0.0 /*time*/);

      // set the coefficients
      shear_modulus_coefficients.set_coefficient_cell(cell, q, shear_modulus_vec);
    }
  }
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress(
  tensor       const & gradient_displacement,
  unsigned int const                                              cell,
  unsigned int const                                              q) const
{
  tensor S;

  if(shear_modulus_is_variable)
  {
	shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  tensor F = get_F<dim, Number>(gradient_displacement);
  scalar J = determinant(F);
  tensor C = transpose(F) * F;
  scalar I_1 = trace(C);
  tensor C_inv = invert(C);

  // S_vol, i.e., penalty term enforcing J = 1.
  S = (data.bulk_modulus * 0.5 * (J*J - 1.0)) * C_inv;

  // S_iso, isochoric term.
  S -= (shear_modulus_stored * pow(J, static_cast<Number>(-2.0 * one_third))) * subtract_identity<dim, Number>(one_third * I_1 * C_inv);

  return S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const & gradient_increment,
  tensor const & deformation_gradient,
  unsigned int const                                              cell,
  unsigned int const                                              q) const
{
  tensor S_lin;

  tensor F = get_F<dim, Number>(gradient_displacement);
  scalar J = determinant(F);
  tensor C = transpose(F) * F;
  scalar I_1 = trace(C);

  tensor F_inv = invert(F);
  tensor C_inv = invert(C);

  scalar J_lin = J * trace(F_inv * )

  // S_vol, i.e., penalty term enforcing J = 1.

  // S_iso, isochoric term.
}

template class IncompressibleNeoHookean<2, float>;
template class IncompressibleNeoHookean<2, double>;

template class IncompressibleNeoHookean<3, float>;
template class IncompressibleNeoHookean<3, double>;

} // namespace Structure
} // namespace ExaDG
