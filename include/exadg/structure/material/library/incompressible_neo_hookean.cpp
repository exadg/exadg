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
template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  IncompressibleNeoHookean(dealii::MatrixFree<dim, Number> const &   matrix_free,
                           unsigned int const                        dof_index,
                           unsigned int const                        quad_index,
                           IncompressibleNeoHookeanData<dim> const & data,
                           bool const                                spatial_integration,
                           bool const                                force_material_residual)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual)
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
    matrix_free.cell_loop(
      &IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
        cell_loop_set_coefficients,
      this,
      dummy,
      dummy);
  }

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement vector.
  if constexpr(cache_level > 0)
  {
    Jm1_coefficients.initialize(matrix_free, quad_index, false, false);
    Jm1_coefficients.set_coefficients(0.0);

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

    if constexpr(cache_level > 1)
    {
      tensor const zero_tensor;
      if(spatial_integration)
      {
        kirchhoff_stress_coefficients.initialize(matrix_free, quad_index, false, false);
        kirchhoff_stress_coefficients.set_coefficients(zero_tensor);

        C_coefficients.initialize(matrix_free, quad_index, false, false);
        C_coefficients.set_coefficients(get_identity<dim, Number>());

        if(force_material_residual)
        {
          gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
          gradient_displacement_coefficients.set_coefficients(zero_tensor);

          second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                                quad_index,
                                                                false,
                                                                false);
          second_piola_kirchhoff_stress_coefficients.set_coefficients(zero_tensor);
        }
      }
      else
      {
        gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
        gradient_displacement_coefficients.set_coefficients(zero_tensor);

        second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                              quad_index,
                                                              false,
                                                              false);
        second_piola_kirchhoff_stress_coefficients.set_coefficients(zero_tensor);

        F_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        F_inv_coefficients.set_coefficients(get_identity<dim, Number>());

        C_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        C_inv_coefficients.set_coefficients(get_identity<dim, Number>());
      }

      AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));
    }
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
void
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
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

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
void
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const
{
  AssertThrow(cache_level > 0 and cache_level < 3,
              dealii::ExcMessage("0 < cache level < 3 expected."));

  for(unsigned int q = 0; q < integrator_lin->n_q_points; ++q)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    tensor Grad_d_lin = integrator_lin->get_gradient(q);

    scalar Jm1;
    tensor F;
    get_modified_F_Jm1<dim, Number, check_type, stable_formulation>(F,
                                                                    Jm1,
                                                                    Grad_d_lin,
                                                                    true /* compute_J */);

    // Overwrite computed values with admissible stored ones
    if constexpr(check_type == 2)
    {
      tensor F_old = gradient_displacement_coefficients.get_coefficient_cell(cell, q);
      add_scaled_identity(F_old, 1.0);

      bool update_J = false;
      for(unsigned int i = 0; i < Jm1.size(); ++i)
      {
        if(Jm1[i] + 1.0 <= get_J_tol<Number>())
        {
          update_J = true;

          for(unsigned int j = 0; j < dim; ++j)
          {
            for(unsigned int k = 0; k < dim; ++k)
            {
              F[j][k][i]          = F_old[j][k][i];
              Grad_d_lin[j][k][i] = F_old[j][k][i];
            }
            Grad_d_lin[j][j][i] -= 1.0;
          }
        }
      }

      if(update_J)
      {
        AssertThrow(stable_formulation == false,
                    dealii::ExcMessage("Storing F_old does not allow for a stable recovery of J."));
        Jm1 = determinant(F) - 1.0;
      }
    }

    Jm1_coefficients.set_coefficient_cell(cell, q, Jm1);

    scalar const J_pow = get_J_pow<true /* force_evaluation */>(Jm1, cell, q);
    J_pow_coefficients.set_coefficient_cell(cell, q, J_pow);

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / (Jm1 + 1.0));
    }

    tensor const E = get_E_scaled<dim, Number, Number, stable_formulation>(Grad_d_lin, 1.0);
    scalar const c1 =
      get_c1<true /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    scalar const c2 =
      get_c2<true /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

    c1_coefficients.set_coefficient_cell(cell, q, c1);
    c2_coefficients.set_coefficient_cell(cell, q, c2);

    if constexpr(cache_level > 1)
    {
      if(spatial_integration)
      {
        tensor const tau = this->kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);

        tensor const C = transpose(F) * F;
        C_coefficients.set_coefficient_cell(cell, q, C);

        if(force_material_residual)
        {
          gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);

          tensor const S =
            this->second_piola_kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
        }
      }
      else
      {
        gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);

        tensor const S =
          this->second_piola_kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);

        tensor const F_inv = invert(F);
        F_inv_coefficients.set_coefficient_cell(cell, q, F_inv);

        tensor const C_inv = F_inv * transpose(F_inv);
        C_inv_coefficients.set_coefficient_cell(cell, q, C_inv);
      }
    }
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::get_c1(
  scalar const &     Jm1,
  scalar const &     J_pow,
  tensor const &     E,
  scalar const &     shear_modulus_stored,
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    return ((0.5 * bulk_modulus) * get_JJm1<Number, stable_formulation>(Jm1) -
         shear_modulus_stored * one_third * J_pow * get_I_1<dim, Number>(E, stable_formulation));
  }
  else
  {
    return c1_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::get_c2(
  scalar const &     Jm1,
  scalar const &     J_pow,
  tensor const &     E,
  scalar const &     shear_modulus_stored,
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    return (bulk_modulus * (get_JJm1<Number, stable_formulation>(Jm1) + 1.0) +
         (2.0 * one_third * one_third) * shear_modulus_stored * J_pow *
           get_I_1<dim, Number>(E, stable_formulation));
  }
  else
  {
    return c2_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
template<bool force_evaluation>
inline dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::get_J_pow(
  scalar const &     Jm1,
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    if constexpr(cache_level == 0)
    {
      // Compute the third root of J^2 via an in-place Newton method.
      // J^2 = 1 is a good initial guess, since we enforce this value
      // implicitly via the constitutive model. We hard-code 3 iterations,
      // which were in most tests enough, but this might be risky and
      // not pay off enough for cache-level > 1 since we are storing
      // this variable anyways.
      scalar J_pow = dealii::make_vectorized_array(static_cast<Number>(1.0));
      scalar J_sqrd = (Jm1 * Jm1 + 2.0 * Jm1 + 1.0);
      J_pow -= (J_pow * J_pow * J_pow - J_sqrd) / (3.0 * J_pow * J_pow);
      J_pow -= (J_pow * J_pow * J_pow - J_sqrd) / (3.0 * J_pow * J_pow);
      J_pow -= (J_pow * J_pow * J_pow - J_sqrd) / (3.0 * J_pow * J_pow);
      return (1.0 / J_pow);
    }
    else
    {
      return (pow(Jm1 + 1.0, static_cast<Number>(-2.0 * one_third)));
    }
  }
  else
  {
    return J_pow_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement_cache_level_0_1,
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

    scalar Jm1;
    tensor F;
    get_modified_F_Jm1<dim, Number, check_type, stable_formulation>(
      F,
      Jm1,
      gradient_displacement_cache_level_0_1,
      cache_level == 0 or force_evaluation /* compute_J */);
    if(cache_level == 1 and (not force_evaluation) and stable_formulation)
    {
      Jm1 = Jm1_coefficients.get_coefficient_cell(cell, q);
    }

    tensor const F_inv = invert(F);
    tensor const C_inv = F_inv * transpose(F_inv);

    scalar J_pow;
    if(force_evaluation)
    {
      J_pow = get_J_pow<true>(Jm1, cell, q);
    }
    else
    {
	  J_pow = get_J_pow<false>(Jm1, cell, q);
    }

    if constexpr(stable_formulation)
    {
      S =
        get_E_scaled<dim, Number, scalar, stable_formulation>(gradient_displacement_cache_level_0_1,
                                                              2.0 * shear_modulus_stored * J_pow);
      add_scaled_identity<dim, Number>(
        S, -one_third * trace(S) + 0.5 * bulk_modulus * get_JJm1<Number, stable_formulation>(Jm1));
      S = C_inv * S;
    }
    else
    {
      if(cache_level == 0 or force_evaluation)
      {
        S = get_E_scaled<dim, Number, Number, stable_formulation>(
          gradient_displacement_cache_level_0_1, 1.0);
      }

      scalar c1;
      if(force_evaluation)
      {
        c1 = get_c1<true>(Jm1, J_pow, S /* E */, shear_modulus_stored, cell, q);
      }
      else
      {
        c1 = get_c1<false>(Jm1, J_pow, S /* E */, shear_modulus_stored, cell, q);
      }

      S = C_inv * c1;
      add_scaled_identity(S, shear_modulus_stored * J_pow);
    }
  }
  else
  {
    S = second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }

  return S;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress_displacement_derivative(
    tensor const &     gradient_increment,
    tensor const &     gradient_displacement_cache_level_0_1,
    unsigned int const cell,
    unsigned int const q) const
{
  tensor Dd_S;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  tensor F;
  if constexpr(cache_level < 2)
  {
    F = gradient_displacement_cache_level_0_1;
  }
  else
  {
    F = gradient_displacement_coefficients.get_coefficient_cell(cell, q);
  }
  add_scaled_identity<dim, Number, Number>(F, 1.0);

  scalar Jm1_cache_level_0;
  tensor E_cache_level_0;
  if constexpr(cache_level == 0)
  {
    get_modified_F_Jm1<dim, Number, check_type, stable_formulation>(
      E_cache_level_0 /* F */,
      Jm1_cache_level_0,
      gradient_displacement_cache_level_0_1,
      true /* compute_J */);

    E_cache_level_0 =
      get_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement_cache_level_0_1,
                                                            1.0);
  }
  else
  {
    // Dummy Jm1 and E sufficient.
  }

  scalar const J_pow = get_J_pow<false /* force_evaluation */>(Jm1_cache_level_0, cell, q);
  scalar const c1    = get_c1<false /* force_evaluation */>(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           cell,
                           q);
  scalar const c2    = get_c2<false /* force_evaluation */>(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           cell,
                           q);

  tensor F_inv, C_inv;
  if constexpr(cache_level < 2)
  {
    F_inv = invert(F);
    C_inv = F_inv * transpose(F_inv);
  }
  else
  {
    F_inv = F_inv_coefficients.get_coefficient_cell(cell, q);
    C_inv = C_inv_coefficients.get_coefficient_cell(cell, q);
  }

  tensor const F_inv_times_gradient_increment = F_inv * gradient_increment;

  scalar const one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
  tensor const Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
  tensor const Dd_C_inv =
    Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

  Dd_S = C_inv * (c2 * one_over_J_times_Dd_J - ((2.0 * one_third) * shear_modulus_stored * J_pow) *
                                                 trace(transpose(gradient_increment) * F));
  Dd_S += Dd_C_inv * c1;
  add_scaled_identity(Dd_S,
                      -shear_modulus_stored * (2.0 * one_third) * J_pow * one_over_J_times_Dd_J);

  return Dd_S;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  kirchhoff_stress(tensor const &     gradient_displacement_cache_level_0_1,
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

    scalar Jm1;
    tensor F;
    get_modified_F_Jm1<dim, Number, check_type, stable_formulation>(
      F,
      Jm1,
      gradient_displacement_cache_level_0_1,
      cache_level == 0 or force_evaluation /* compute_J */);
    if(cache_level == 1 and (not force_evaluation) and stable_formulation)
    {
      Jm1 = Jm1_coefficients.get_coefficient_cell(cell, q);
    }

    scalar J_pow;
    if(force_evaluation)
    {
      J_pow = get_J_pow<true>(Jm1, cell, q);
    }
    else
    {
      J_pow = get_J_pow<false>(Jm1, cell, q);
    }

    if constexpr(stable_formulation)
    {
      tau =
        get_E_scaled<dim, Number, scalar, stable_formulation>(gradient_displacement_cache_level_0_1,
                                                              2.0 * shear_modulus_stored * J_pow);
      add_scaled_identity<dim, Number>(tau,
                                       -one_third * trace(tau) +
                                         0.5 * bulk_modulus *
                                           get_JJm1<Number, stable_formulation>(Jm1));
    }
    else
    {
      if(cache_level == 0 or force_evaluation)
      {
        tau = get_E_scaled<dim, Number, Number, stable_formulation>(
          gradient_displacement_cache_level_0_1, 1.0);
      }
      else
      {
        // Dummy E sufficient.
      }

      scalar c1;
      if(force_evaluation)
      {
        c1 = get_c1<true>(Jm1, J_pow, tau /* E */, shear_modulus_stored, cell, q);
      }
      else
      {
	    c1 = get_c1<false>(Jm1, J_pow, tau /* E */, shear_modulus_stored, cell, q);
      }

      tau = (F * transpose(F)) * (shear_modulus_stored * J_pow);
      add_scaled_identity(tau, c1);
    }
  }
  else
  {
    tau = kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(tensor const &     symmetric_gradient_increment,
                          tensor const &     gradient_displacement_cache_level_0_1,
                          unsigned int const cell,
                          unsigned int const q) const
{
  tensor result;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  tensor C;
  if constexpr(cache_level < 2)
  {
    tensor F = gradient_displacement_cache_level_0_1;
    add_scaled_identity<dim, Number, Number>(F, 1.0);
    C = transpose(F) * F;
  }
  else
  {
    C = C_coefficients.get_coefficient_cell(cell, q);
  }

  scalar Jm1_cache_level_0;
  tensor E_cache_level_0;
  if constexpr(cache_level == 0)
  {
    get_modified_F_Jm1<dim, Number, check_type, stable_formulation>(
      result, Jm1_cache_level_0, gradient_displacement_cache_level_0_1, true /* compute_J */);

    E_cache_level_0 =
      get_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement_cache_level_0_1,
                                                            1.0);
  }
  else
  {
    // Dummy E and Jm1 sufficient.
  }

  scalar const J_pow = get_J_pow<false /* force_evaluation */>(Jm1_cache_level_0, cell, q);
  scalar const c1    = get_c1<false /* force_evaluation */>(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           cell,
                           q);
  scalar const c2    = get_c2<false /* force_evaluation */>(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           cell,
                           q);

  result = symmetric_gradient_increment * (-2.0 * c1);
  result +=
    ((-4.0 * one_third) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) * C;
  add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

  return result;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::one_over_J(
  unsigned int const cell,
  unsigned int const q) const
{
  AssertThrow(spatial_integration and cache_level > 0,
              dealii::ExcMessage("Cannot access precomputed one_over_J."));
  return (one_over_J_coefficients.get_coefficient_cell(cell, q));
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  gradient_displacement(unsigned int const cell, unsigned int const q) const
{
  AssertThrow(cache_level > 1,
              dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  return (gradient_displacement_coefficients.get_coefficient_cell(cell, q));
}

// clang-format off
// Note that the higher check types (third template argument) are missing.
template class IncompressibleNeoHookean<2, float,  0, true,  0>;
template class IncompressibleNeoHookean<2, float,  0, true,  1>;
template class IncompressibleNeoHookean<2, float,  0, true,  2>;

template class IncompressibleNeoHookean<2, float,  0, false, 0>;
template class IncompressibleNeoHookean<2, float,  0, false, 1>;
template class IncompressibleNeoHookean<2, float,  0, false, 2>;

template class IncompressibleNeoHookean<2, double, 0, true,  0>;
template class IncompressibleNeoHookean<2, double, 0, true,  1>;
template class IncompressibleNeoHookean<2, double, 0, true,  2>;

template class IncompressibleNeoHookean<2, double, 0, false, 0>;
template class IncompressibleNeoHookean<2, double, 0, false, 1>;
template class IncompressibleNeoHookean<2, double, 0, false, 2>;

template class IncompressibleNeoHookean<3, float,  0, true,  0>;
template class IncompressibleNeoHookean<3, float,  0, true,  1>;
template class IncompressibleNeoHookean<3, float,  0, true,  2>;

template class IncompressibleNeoHookean<3, float,  0, false, 0>;
template class IncompressibleNeoHookean<3, float,  0, false, 1>;
template class IncompressibleNeoHookean<3, float,  0, false, 2>;

template class IncompressibleNeoHookean<3, double, 0, true,  0>;
template class IncompressibleNeoHookean<3, double, 0, true,  1>;
template class IncompressibleNeoHookean<3, double, 0, true,  2>;

template class IncompressibleNeoHookean<3, double, 0, false, 0>;
template class IncompressibleNeoHookean<3, double, 0, false, 1>;
template class IncompressibleNeoHookean<3, double, 0, false, 2>;
// clang-format on

} // namespace Structure
} // namespace ExaDG
