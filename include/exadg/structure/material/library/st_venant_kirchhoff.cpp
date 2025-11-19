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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
StVenantKirchhoff<dim, Number>::StVenantKirchhoff(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  StVenantKirchhoffData<dim> const &      data,
  bool const                              large_deformation)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    large_deformation(large_deformation),
    f0_factor(get_f0_factor(data.poissons_ratio, data.type_two_dim)),
    f1_factor(get_f1_factor(data.poissons_ratio, data.type_two_dim)),
    f2_factor(get_f2_factor(data.poissons_ratio, data.type_two_dim)),
    youngs_modulus_is_variable(data.youngs_modulus_function != nullptr)
{
  // initialize factors, overwritten using stored coefficients in variable coefficients case, i.e.,
  // when `youngs_modulus_is_variable == true`.
  f0 = dealii::make_vectorized_array<Number>(f0_factor * data.youngs_modulus);
  f1 = dealii::make_vectorized_array<Number>(f1_factor * data.youngs_modulus);
  f2 = dealii::make_vectorized_array<Number>(f2_factor * data.youngs_modulus);

  if(youngs_modulus_is_variable)
  {
    // allocate vectors for variable coefficients
    f0_coefficients.initialize(matrix_free, quad_index, false, false);
    f1_coefficients.initialize(matrix_free, quad_index, false, false);
    f2_coefficients.initialize(matrix_free, quad_index, false, false);

    VectorType dummy;
    matrix_free.cell_loop(&StVenantKirchhoff<dim, Number>::cell_loop_set_coefficients,
                          this,
                          dummy,
                          dummy);
  }
}

template<int dim, typename Number>
Number
StVenantKirchhoff<dim, Number>::get_f0_factor(Number const & poissons_ratio,
                                              Type2D const   type_two_dim) const
{
  if constexpr(dim == 3)
  {
    return (1. - poissons_ratio) / ((1. + poissons_ratio) * (1. - 2. * poissons_ratio));
  }
  else
  {
    if(type_two_dim == Type2D::PlaneStress)
    {
      return (1. / (1. - poissons_ratio * poissons_ratio));
    }
    else
    {
      return ((1. - poissons_ratio) / ((1. + poissons_ratio) * (1. - 2. * poissons_ratio)));
    }
  }
}

template<int dim, typename Number>
Number
StVenantKirchhoff<dim, Number>::get_f1_factor(Number const & poissons_ratio,
                                              Type2D const   type_two_dim) const
{
  if constexpr(dim == 3)
  {
    return (poissons_ratio / ((1. + poissons_ratio) * (1. - 2. * poissons_ratio)));
  }
  else
  {
    if(type_two_dim == Type2D::PlaneStress)
    {
      return (poissons_ratio / (1. - poissons_ratio * poissons_ratio));
    }
    else
    {
      return (poissons_ratio / ((1. + poissons_ratio) * (1. - 2. * poissons_ratio)));
    }
  }
}

template<int dim, typename Number>
Number
StVenantKirchhoff<dim, Number>::get_f2_factor(Number const & poissons_ratio,
                                              Type2D const   type_two_dim) const
{
  if constexpr(dim == 3)
  {
    return ((1. - 2. * poissons_ratio) /
            (2.0 * (1. + poissons_ratio) * (1. - 2. * poissons_ratio)));
  }
  else
  {
    if(type_two_dim == Type2D::PlaneStress)
    {
      return ((1. - poissons_ratio) / (2.0 * (1. - poissons_ratio * poissons_ratio)));
    }
    else
    {
      return ((1. - 2. * poissons_ratio) /
              (2.0 * (1. + poissons_ratio) * (1. - 2. * poissons_ratio)));
    }
  }
}

template<int dim, typename Number>
void
StVenantKirchhoff<dim, Number>::cell_loop_set_coefficients(
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
      dealii::VectorizedArray<Number> youngs_modulus_vec =
        FunctionEvaluator<0, dim, Number>::value(*(data.youngs_modulus_function),
                                                 integrator.quadrature_point(q),
                                                 0.0 /*time*/);

      // set the coefficients
      f0_coefficients.set_coefficient_cell(cell, q, f0_factor * youngs_modulus_vec);
      f1_coefficients.set_coefficient_cell(cell, q, f1_factor * youngs_modulus_vec);
      f2_coefficients.set_coefficient_cell(cell, q, f2_factor * youngs_modulus_vec);
    }
  }
}

template<int dim, typename Number>
typename StVenantKirchhoff<dim, Number>::symmetric_tensor
StVenantKirchhoff<dim, Number>::second_piola_kirchhoff_stress_symmetrize(tensor const &     strain,
                                                                         unsigned int const cell,
                                                                         unsigned int const q) const
{
  symmetric_tensor S;

  if(youngs_modulus_is_variable)
  {
    f0 = f0_coefficients.get_coefficient_cell(cell, q);
    f1 = f1_coefficients.get_coefficient_cell(cell, q);
    f2 = f2_coefficients.get_coefficient_cell(cell, q);
  }

  // Since we return a `symmetric_tensor`, we only need to set symmetric entries once.
  if constexpr(dim == 3)
  {
    S[0][0] = f0 * strain[0][0] + f1 * (strain[1][1] + strain[2][2]);
    S[1][1] = f0 * strain[1][1] + f1 * (strain[0][0] + strain[2][2]);
    S[2][2] = f0 * strain[2][2] + f1 * (strain[0][0] + strain[1][1]);
    S[0][1] = f2 * (strain[0][1] + strain[1][0]);
    S[1][2] = f2 * (strain[1][2] + strain[2][1]);
    S[0][2] = f2 * (strain[0][2] + strain[2][0]);
  }
  else
  {
    S[0][0] = f0 * strain[0][0] + f1 * strain[1][1];
    S[1][1] = f1 * strain[0][0] + f0 * strain[1][1];
    S[0][1] = f2 * (strain[0][1] + strain[1][0]);
  }

  return S;
}

template<int dim, typename Number>
typename StVenantKirchhoff<dim, Number>::symmetric_tensor
StVenantKirchhoff<dim, Number>::second_piola_kirchhoff_stress(tensor const & gradient_displacement,
                                                              unsigned int const cell,
                                                              unsigned int const q) const
{
  if(large_deformation)
  {
    return (this->second_piola_kirchhoff_stress_symmetrize(
      get_E<dim, Number>(gradient_displacement), cell, q));
  }
  else
  {
    return (this->second_piola_kirchhoff_stress_symmetrize(gradient_displacement, cell, q));
  }
}

template<int dim, typename Number>
typename StVenantKirchhoff<dim, Number>::symmetric_tensor
StVenantKirchhoff<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  // Exploit linear stress-strain relationship and symmetrizing in
  // `second_piola_kirchhoff_stress_symmetrize()`.
  return (this->second_piola_kirchhoff_stress_symmetrize(
    transpose(deformation_gradient) * gradient_increment, cell, q));
}

template class StVenantKirchhoff<2, float>;
template class StVenantKirchhoff<2, double>;

template class StVenantKirchhoff<3, float>;
template class StVenantKirchhoff<3, double>;

} // namespace Structure
} // namespace ExaDG
