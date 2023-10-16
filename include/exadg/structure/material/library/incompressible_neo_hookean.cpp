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
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  unsigned int const                        dof_index,
  unsigned int const                        quad_index,
  IncompressibleNeoHookeanData<dim> const & data,
  bool const                                spatial_integration,
  bool const                                force_material_residual,
  unsigned int const                        cache_level)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual),
    check_type(0),
    cache_level(cache_level)
{
  // initialize (potentially variable) shear modulus
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored       = dealii::make_vectorized_array<Number>(shear_modulus);

  bulk_modulus = static_cast<Number>(data.bulk_modulus);

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

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement vector.
  if(cache_level > 0)
  {
    J_pow_coefficients.initialize(matrix_free, quad_index, false, false);
    J_pow_coefficients.set_coefficients(1.0);

    c1_coefficients.initialize(matrix_free, quad_index, false, false);
    c1_coefficients.set_coefficients(-shear_modulus * one_third * static_cast<Number>(dim));
    c2_coefficients.initialize(matrix_free, quad_index, false, false);
    c2_coefficients.set_coefficients(
      shear_modulus * one_third * 2.0 * one_third * static_cast<Number>(dim) + bulk_modulus);

    if(spatial_integration)
    {
      one_over_J_coefficients.initialize(matrix_free, quad_index, false, false);
      one_over_J_coefficients.set_coefficients(1.0);
    }

    if(cache_level > 1)
    {
      deformation_gradient_coefficients.initialize(matrix_free, quad_index, false, false);
      deformation_gradient_coefficients.set_coefficients(get_identity<dim, Number>());

      tensor const zero_tensor;
      if(spatial_integration)
      {
        kirchhoff_stress_coefficients.initialize(matrix_free, quad_index, false, false);
        kirchhoff_stress_coefficients.set_coefficients(zero_tensor);

        C_coefficients.initialize(matrix_free, quad_index, false, false);
        C_coefficients.set_coefficients(get_identity<dim, Number>());
      }
      else
      {
        F_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        F_inv_coefficients.set_coefficients(get_identity<dim, Number>());

        C_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        C_inv_coefficients.set_coefficients(get_identity<dim, Number>());
      }

      if(force_material_residual or not spatial_integration)
      {
        second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                              quad_index,
                                                              false,
                                                              false);
        second_piola_kirchhoff_stress_coefficients.set_coefficients(zero_tensor);
      }

      AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));
    }
  }
}

template<int dim, typename Number>
void
IncompressibleNeoHookean<dim, Number>::do_set_cell_linearization_data(
  std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
  unsigned int const                                                         cell) const
{
  AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));

  for(unsigned int q = 0; q < integrator_lin->n_q_points; ++q)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    tensor const Grad_d_lin = integrator_lin->get_gradient(q);

    scalar J;
    tensor F;
    get_modified_F_J(F, J, Grad_d_lin, check_type, true /* compute_J */);

    scalar const J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
    J_pow_coefficients.set_coefficient_cell(cell, q, J_pow);

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / J);
    }

    tensor const C   = transpose(F) * F;
    scalar const I_1 = trace(C);
    scalar const c1 =
      -shear_modulus_stored * one_third * J_pow * I_1 + bulk_modulus * 0.5 * (J * J - 1.0);
    c1_coefficients.set_coefficient_cell(cell, q, c1);

    scalar const c2 =
      shear_modulus_stored * one_third * one_third * 2.0 * J_pow * I_1 + bulk_modulus * J * J;
    c2_coefficients.set_coefficient_cell(cell, q, c2);

    if(cache_level > 1)
    {
      deformation_gradient_coefficients.set_coefficient_cell(cell, q, F);

      if(spatial_integration)
      {
        tensor const tau_lin =
          this->kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau_lin);

        C_coefficients.set_coefficient_cell(cell, q, C);
      }
      else
      {
        tensor const F_inv = invert(F);
        F_inv_coefficients.set_coefficient_cell(cell, q, F_inv);

        tensor const C_inv = F_inv * transpose(F_inv);
        C_inv_coefficients.set_coefficient_cell(cell, q, C_inv);
      }

      if(force_material_residual or not spatial_integration)
      {
        tensor const S_lin =
          this->second_piola_kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S_lin);
      }
    }
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
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q,
  bool const         force_evaluation) const
{
  tensor S;
  if(cache_level < 2 or force_evaluation)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    scalar J;
    tensor F;
    get_modified_F_J(F, J, gradient_displacement, check_type, false /* compute_J */);

    tensor const C = transpose(F) * F;

    scalar J_pow, c1;
    if(cache_level == 0)
    {
      J = determinant(F);

      J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
      c1 =
        (-shear_modulus_stored * J_pow * one_third * trace(C) + bulk_modulus * 0.5 * (J * J - 1.0));
    }
    else
    {
      J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      c1    = c1_coefficients.get_coefficient_cell(cell, q);
    }

    S = invert(C) * c1;
    add_scaled_identity(S, shear_modulus_stored * J_pow);
  }
  else
  {
    S = second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }

  return S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  tensor Dd_S;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  scalar J_pow, c1, c2;
  if(cache_level == 0)
  {
    scalar const J   = determinant(deformation_gradient);
    tensor const C   = transpose(deformation_gradient) * deformation_gradient;
    scalar const I_1 = trace(C);

    J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
    c1    = -shear_modulus_stored * one_third * J_pow * I_1 + bulk_modulus * 0.5 * (J * J - 1.0);
    c2    = shear_modulus_stored * one_third * J_pow * 2.0 * one_third * I_1 + bulk_modulus * J * J;
  }
  else
  {
    J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    c1    = c1_coefficients.get_coefficient_cell(cell, q);
    c2    = c2_coefficients.get_coefficient_cell(cell, q);
  }

  tensor F_inv, C_inv;
  if(cache_level < 2)
  {
    F_inv = invert(deformation_gradient);
    C_inv = F_inv * transpose(F_inv);
  }
  else
  {
    F_inv = F_inv_coefficients.get_coefficient_cell(cell, q);
    C_inv = C_inv_coefficients.get_coefficient_cell(cell, q);
  }

  scalar const one_over_J_times_Dd_J = trace(F_inv * gradient_increment);
  scalar const Dd_I_1 = 2.0 * trace(transpose(gradient_increment) * deformation_gradient);
  tensor const Dd_F_inv_times_transpose_F_inv = -F_inv * gradient_increment * C_inv;
  tensor const Dd_C_inv =
    Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

  Dd_S =
    C_inv * (Dd_I_1 * (-shear_modulus_stored * one_third * J_pow) + one_over_J_times_Dd_J * c2);
  Dd_S += Dd_C_inv * c1;
  add_scaled_identity(Dd_S,
                      -shear_modulus_stored * one_third * J_pow * 2.0 * one_over_J_times_Dd_J);

  return Dd_S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q,
                                                        bool const         force_evaluation) const
{
  tensor tau;
  if(cache_level < 2 or force_evaluation)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    scalar J;
    tensor F;
    get_modified_F_J(F, J, gradient_displacement, check_type, false /* compute_J */);

    tensor const F_times_F_transposed = F * transpose(F);

    scalar J_pow, c1;
    if(cache_level == 0)
    {
      J = determinant(F);

      J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
      c1    = (-shear_modulus_stored * J_pow * one_third * trace(F_times_F_transposed) /* = I_1 */
            + bulk_modulus * 0.5 * (J * J - 1.0));
    }
    else
    {
      J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      c1    = c1_coefficients.get_coefficient_cell(cell, q);
    }

    tau = F_times_F_transposed * (shear_modulus_stored * J_pow);
    add_scaled_identity(tau, c1);
  }
  else
  {
    tau = kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }

  return tau;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::contract_with_J_times_C(
  tensor const &     symmetric_gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  tensor result;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  tensor C;
  if(cache_level < 2)
  {
    C = transpose(deformation_gradient) * deformation_gradient;
  }
  else
  {
    C = C_coefficients.get_coefficient_cell(cell, q);
  }

  scalar J_pow, c1, c2;
  if(cache_level == 0)
  {
    scalar const I_1 = trace(C);
    scalar const J   = determinant(deformation_gradient);

    J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
    c1    = -shear_modulus_stored * J_pow * one_third * I_1 + bulk_modulus * 0.5 * (J * J - 1.0);
    c2    = shear_modulus_stored * one_third * J_pow * 2.0 * one_third * I_1 + bulk_modulus * J * J;
  }
  else
  {
    J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    c1    = c1_coefficients.get_coefficient_cell(cell, q);
    c2    = c2_coefficients.get_coefficient_cell(cell, q);
  }

  result              = symmetric_gradient_increment * (-2.0 * c1);
  scalar const factor = -4.0 * one_third * shear_modulus_stored * J_pow *
                          scalar_product(C, symmetric_gradient_increment) +
                        c2 * trace(symmetric_gradient_increment);
  add_scaled_identity(result, factor);

  return result;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number>::one_over_J(unsigned int const cell,
                                                  unsigned int const q) const
{
  AssertThrow(spatial_integration and cache_level > 0,
              dealii::ExcMessage("Cannot access precomputed one_over_J."));
  return (one_over_J_coefficients.get_coefficient_cell(cell, q));
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number>::deformation_gradient(unsigned int const cell,
                                                            unsigned int const q) const
{
  AssertThrow(cache_level > 1,
              dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  return (deformation_gradient_coefficients.get_coefficient_cell(cell, q));
}

template class IncompressibleNeoHookean<2, float>;
template class IncompressibleNeoHookean<2, double>;

template class IncompressibleNeoHookean<3, float>;
template class IncompressibleNeoHookean<3, double>;

} // namespace Structure
} // namespace ExaDG
