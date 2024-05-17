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
    bulk_modulus(static_cast<Number>(data.bulk_modulus)),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual)
{
  // initialize (potentially variable) shear modulus
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored       = dealii::make_vectorized_array<Number>(shear_modulus);

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
    c1_coefficients.set_coefficients(-shear_modulus * ONE_THIRD * static_cast<Number>(dim));
    c2_coefficients.initialize(matrix_free, quad_index, false, false);
    c2_coefficients.set_coefficients(shear_modulus * TWO_NINTHS * static_cast<Number>(dim) +
                                     bulk_modulus);

    if(spatial_integration)
    {
      one_over_J_coefficients.initialize(matrix_free, quad_index, false, false);
      one_over_J_coefficients.set_coefficients(1.0);
    }

    if constexpr(cache_level > 1)
    {
      if(spatial_integration)
      {
        kirchhoff_stress_coefficients.initialize(matrix_free, quad_index, false, false);
        kirchhoff_stress_coefficients.set_coefficients(get_zero_symmetric_tensor<dim, Number>());

        C_coefficients.initialize(matrix_free, quad_index, false, false);
        C_coefficients.set_coefficients(get_identity_symmetric_tensor<dim, Number>());

        if(force_material_residual)
        {
          gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
          gradient_displacement_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

          second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                                quad_index,
                                                                false,
                                                                false);
          second_piola_kirchhoff_stress_coefficients.set_coefficients(
            get_zero_symmetric_tensor<dim, Number>());
        }
      }
      else
      {
        gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
        gradient_displacement_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

        F_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        F_inv_coefficients.set_coefficients(get_identity_tensor<dim, Number>());

        C_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        C_inv_coefficients.set_coefficients(get_identity_symmetric_tensor<dim, Number>());

        second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                              quad_index,
                                                              false,
                                                              false);
        second_piola_kirchhoff_stress_coefficients.set_coefficients(
          get_zero_symmetric_tensor<dim, Number>());
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
    auto [F, Jm1] = compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(Grad_d_lin);

    // Overwrite computed values with admissible stored ones
    if constexpr(check_type == 2)
    {
      tensor F_old = gradient_displacement_coefficients.get_coefficient_cell(cell, q);
      add_scaled_identity(F_old, static_cast<Number>(1.0));

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

    symmetric_tensor const E =
      compute_E_scaled<dim, Number, Number, stable_formulation>(Grad_d_lin, 1.0);
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
        if constexpr(stable_formulation)
        {
          symmetric_tensor const tau =
            compute_tau_stable(Grad_d_lin, Jm1, J_pow, shear_modulus_stored);
          kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);
        }
        else
        {
          symmetric_tensor const tau = compute_tau_unstable(F, J_pow, c1, shear_modulus_stored);
          kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);
        }

        symmetric_tensor const C = compute_HT_times_H(F);
        C_coefficients.set_coefficient_cell(cell, q, C);

        if(force_material_residual)
        {
          gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);

          tensor const           F_inv = invert(F);
          symmetric_tensor const C_inv = compute_C_inv(F_inv);
          if constexpr(stable_formulation)
          {
            symmetric_tensor const S =
              compute_S_stable(Grad_d_lin, C_inv, J_pow, Jm1, shear_modulus_stored);
            second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
          }
          else
          {
            symmetric_tensor const S = compute_S_unstable(C_inv, J_pow, c1, shear_modulus_stored);
            second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
          }
        }
      }
      else
      {
        gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);

        tensor const F_inv = invert(F);
        F_inv_coefficients.set_coefficient_cell(cell, q, F_inv);

        symmetric_tensor const C_inv = compute_C_inv(F_inv);
        C_inv_coefficients.set_coefficient_cell(cell, q, C_inv);

        if constexpr(stable_formulation)
        {
          symmetric_tensor const S =
            compute_S_stable(Grad_d_lin, C_inv, J_pow, Jm1, shear_modulus_stored);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
        }
        else
        {
          symmetric_tensor const S = compute_S_unstable(C_inv, J_pow, c1, shear_modulus_stored);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
        }
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
  scalar const &           Jm1,
  scalar const &           J_pow,
  symmetric_tensor const & E,
  scalar const &           shear_modulus,
  unsigned int const       cell,
  unsigned int const       q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    return ((0.5 * bulk_modulus) * compute_JJm1<Number, stable_formulation>(Jm1) -
            shear_modulus * ONE_THIRD * J_pow * compute_I_1<dim, Number>(E, stable_formulation));
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
  scalar const &           Jm1,
  scalar const &           J_pow,
  symmetric_tensor const & E,
  scalar const &           shear_modulus,
  unsigned int const       cell,
  unsigned int const       q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    return (bulk_modulus * (compute_JJm1<Number, stable_formulation>(Jm1) + 1.0) +
            TWO_NINTHS * shear_modulus * J_pow * compute_I_1<dim, Number>(E, stable_formulation));
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
      // Compute the inverse third root of J^2 via Newton's method:
      //
      // f(x) = x - J^(-2/3) = 0
      // or equivalently
      // f(x) = x^(-3) - J^2 = 0
      // with
      // f'(x) = - 3 x^(-4)
      // and hence the Newton update
      // x_np1 = 4/3 * x_n - J^2/3 * x_n^4.
      //
      // J^(-2/3) = 1 is a good initial guess, since we enforce J = 1
      // implicitly via the constitutive model. We hard-code 3 steps,
      // which were in most tests enough, but might be risky and hence
      // not pay off enough for cache_level > 1 since we are storing
      // related variables anyways.

      scalar const J_sqrd_over_three = (Jm1 * Jm1 + 2.0 * Jm1 + 1.0) * ONE_THIRD;

      // The first iteration simplifies due to the initial guess x_0 = 1.0.
      scalar J_pow = dealii::make_vectorized_array(static_cast<Number>(FOUR_THIRDS));
      J_pow -= J_sqrd_over_three;

      // The remaining Newton iterations are identical.
      scalar J_pow_to_the_fourth = J_pow * J_pow;
      J_pow_to_the_fourth *= J_pow_to_the_fourth;
      J_pow = FOUR_THIRDS * J_pow - J_sqrd_over_three * J_pow_to_the_fourth;

      J_pow_to_the_fourth = J_pow * J_pow;
      J_pow_to_the_fourth *= J_pow_to_the_fourth;
      J_pow = FOUR_THIRDS * J_pow - J_sqrd_over_three * J_pow_to_the_fourth;

      return J_pow;
    }
    else
    {
      return fast_approx_powp1<Number, stable_formulation>(Jm1, static_cast<Number>(-TWO_THIRDS));
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
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    if constexpr(cache_level == 0)
    {
      auto const [F, Jm1] =
        compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
      tensor const           F_inv = invert(F);
      symmetric_tensor const C_inv = compute_C_inv(F_inv);
      scalar const           J_pow = get_J_pow<false /* force_evaluation */>(Jm1, cell, q);
      if constexpr(stable_formulation)
      {
        return compute_S_stable(gradient_displacement, C_inv, J_pow, Jm1, shear_modulus_stored);
      }
      else
      {
        symmetric_tensor const E =
          compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
        scalar const c1 =
          get_c1<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

        return compute_S_unstable(C_inv, J_pow, c1, shear_modulus_stored);
      }
    }
    else
    {
      tensor const F =
        compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
      tensor const           F_inv = invert(F);
      symmetric_tensor const C_inv = compute_C_inv(F_inv);
      scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      if constexpr(stable_formulation)
      {
        scalar const Jm1 = Jm1_coefficients.get_coefficient_cell(cell, q);
        return compute_S_stable(gradient_displacement, C_inv, J_pow, Jm1, shear_modulus_stored);
      }
      else
      {
        scalar const c1 = c1_coefficients.get_coefficient_cell(cell, q);

        return compute_S_unstable(C_inv, J_pow, c1, shear_modulus_stored);
      }
    }
  }
  else
  {
    AssertThrow(cache_level < 2,
                dealii::ExcMessage("This `cache_level` stores tensorial quantities, "
                                   "use the dedicated function."));

    return second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                   "this `cache_level` does not store tensorial quantities."));

    return (std::numeric_limits<Number>::quiet_NaN() *
            get_identity_symmetric_tensor<dim, Number>());
  }
  else
  {
    return second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_S_stable(tensor const &           gradient_displacement,
                   symmetric_tensor const & C_inv,
                   scalar const &           J_pow,
                   scalar const &           Jm1,
                   scalar const &           shear_modulus) const
{
  symmetric_tensor S =
    compute_E_scaled<dim, Number, scalar, stable_formulation>(gradient_displacement,
                                                              2.0 * shear_modulus * J_pow);
  add_scaled_identity(S,
                      -ONE_THIRD * trace(S) +
                        (0.5 * bulk_modulus) * compute_JJm1<Number, stable_formulation>(Jm1));

  return compute_symmetric_product(C_inv, S);
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_S_unstable(symmetric_tensor const & C_inv,
                     scalar const &           J_pow,
                     scalar const &           c1,
                     scalar const &           shear_modulus) const
{
  symmetric_tensor S = c1 * C_inv;

  add_scaled_identity(S, shear_modulus * J_pow);

  return S;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level == 0)
  {
    auto const [F, Jm1] =
      compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
    symmetric_tensor const E =
      compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
    scalar const J_pow = get_J_pow<false /* force_evaluation */>(Jm1, cell, q);
    scalar const c1 =
      get_c1<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    scalar const c2 =
      get_c2<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    tensor const           F_inv = invert(F);
    symmetric_tensor const C_inv = compute_C_inv(F_inv);

    tensor const           F_inv_times_gradient_increment = F_inv * gradient_increment;
    scalar const           one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
    tensor const           Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
    symmetric_tensor const Dd_C_inv = compute_H_plus_HT(Dd_F_inv_times_transpose_F_inv);

    symmetric_tensor Dd_S =
      C_inv * (c2 * one_over_J_times_Dd_J - (TWO_THIRDS * shear_modulus_stored * J_pow) *
                                              trace(transpose(gradient_increment) * F));
    Dd_S += Dd_C_inv * c1;
    add_scaled_identity(Dd_S, -shear_modulus_stored * TWO_THIRDS * J_pow * one_over_J_times_Dd_J);

    return Dd_S;
  }
  else if constexpr(cache_level == 1)
  {
    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);
    tensor const           F_inv = invert(F);
    symmetric_tensor const C_inv = compute_C_inv(F_inv);
    tensor const           F_inv_times_gradient_increment = F_inv * gradient_increment;
    scalar const           one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
    tensor const           Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
    symmetric_tensor const Dd_C_inv = compute_H_times_HT(Dd_F_inv_times_transpose_F_inv);

    symmetric_tensor Dd_S =
      C_inv * (c2 * one_over_J_times_Dd_J - (TWO_THIRDS * shear_modulus_stored * J_pow) *
                                              trace(transpose(gradient_increment) * F));
    Dd_S += Dd_C_inv * c1;
    add_scaled_identity(Dd_S, -shear_modulus_stored * TWO_THIRDS * J_pow * one_over_J_times_Dd_J);

    return Dd_S;
  }
  else
  {
    // Note that we could load F, but F is needed in the nonlinear operator regardless,
    // so we pass it in and use it here instead of loading it twice and not passing it.
    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);
    tensor const           F_inv = F_inv_coefficients.get_coefficient_cell(cell, q);
    symmetric_tensor const C_inv = C_inv_coefficients.get_coefficient_cell(cell, q);

    tensor const           F_inv_times_gradient_increment = F_inv * gradient_increment;
    scalar const           one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
    tensor const           Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
    symmetric_tensor const Dd_C_inv = compute_H_times_HT(Dd_F_inv_times_transpose_F_inv);

    symmetric_tensor Dd_S =
      C_inv * (c2 * one_over_J_times_Dd_J - (TWO_THIRDS * shear_modulus_stored * J_pow) *
                                              trace(transpose(gradient_increment) * F));
    Dd_S += Dd_C_inv * c1;
    add_scaled_identity(Dd_S, -shear_modulus_stored * TWO_THIRDS * J_pow * one_over_J_times_Dd_J);

    return Dd_S;
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    if(shear_modulus_is_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    }

    if constexpr(cache_level == 0)
    {
      if constexpr(stable_formulation)
      {
        scalar const Jm1 =
          compute_modified_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
        scalar const J_pow = get_J_pow<false /* force_evaluation*/>(Jm1, cell, q);

        return compute_tau_stable(gradient_displacement, Jm1, J_pow, shear_modulus_stored);
      }
      else
      {
        auto const [F, Jm1] = compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(
          gradient_displacement);
        scalar const           J_pow = get_J_pow<false /* force_evaluation*/>(Jm1, cell, q);
        symmetric_tensor const E =
          compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
        scalar const c1 =
          get_c1<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

        return compute_tau_unstable(F, J_pow, c1, shear_modulus_stored);
      }
    }
    else
    {
      if constexpr(stable_formulation)
      {
        scalar const Jm1   = Jm1_coefficients.get_coefficient_cell(cell, q);
        scalar const J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);

        return compute_tau_stable(gradient_displacement, Jm1, J_pow, shear_modulus_stored);
      }
      else
      {
        tensor const F =
          compute_modified_F<dim, Number, cache_level, stable_formulation>(gradient_displacement);
        scalar const J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
        scalar const c1    = c1_coefficients.get_coefficient_cell(cell, q);

        return compute_tau_unstable(F, J_pow, c1, shear_modulus_stored);
      }
    }
  }
  else
  {
    AssertThrow(cache_level < 2,
                dealii::ExcMessage("This `cache_level` stores tensorial quantities, "
                                   "use the dedicated function."));

    return kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                   "this `cache_level` does not store tensorial quantities."));

    return (std::numeric_limits<Number>::quiet_NaN() *
            get_identity_symmetric_tensor<dim, Number>());
  }
  else
  {
    return kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_tau_stable(tensor const & gradient_displacement,
                     scalar const & Jm1,
                     scalar const & J_pow,
                     scalar const & shear_modulus) const
{
  symmetric_tensor tau =
    compute_E_scaled<dim, Number, scalar, stable_formulation>(gradient_displacement,
                                                              2.0 * shear_modulus * J_pow);

  add_scaled_identity(tau,
                      -ONE_THIRD * trace(tau) +
                        (0.5 * bulk_modulus) * compute_JJm1<Number, stable_formulation>(Jm1));

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_tau_unstable(tensor const & F,
                       scalar const & J_pow,
                       scalar const & c1,
                       scalar const & shear_modulus) const
{
  symmetric_tensor tau = (shear_modulus * J_pow) * compute_H_times_HT(F);

  add_scaled_identity(tau, c1);

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level == 0)
  {
    auto const [F, Jm1] =
      compute_modified_F_Jm1<dim, Number, cache_level, stable_formulation>(gradient_displacement);
    symmetric_tensor const E =
      compute_E_scaled<dim, Number, Number, stable_formulation>(gradient_displacement, 1.0);
    scalar const J_pow = get_J_pow<false /* force_evaluation */>(Jm1, cell, q);
    scalar const c1 =
      get_c1<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);
    scalar const c2 =
      get_c2<false /* force_evaluation */>(Jm1, J_pow, E, shear_modulus_stored, cell, q);

    symmetric_tensor result = symmetric_gradient_increment * (-2.0 * c1);
    result +=
      ((-4.0 * ONE_THIRD) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) *
      compute_HT_times_H(F);
    add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

    return result;
  }
  else if constexpr(cache_level == 1)
  {
    tensor const F =
      compute_modified_F<dim, Number, cache_level, stable_formulation>(gradient_displacement);
    scalar const J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const c2    = c2_coefficients.get_coefficient_cell(cell, q);

    symmetric_tensor result = symmetric_gradient_increment * (-2.0 * c1);
    result +=
      ((-4.0 * ONE_THIRD) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) *
      compute_HT_times_H(F);
    add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

    return result;
  }
  else
  {
    AssertThrow(cache_level < 2,
                dealii::ExcMessage("This function is not the optimal choice for `cache_level` 2."));

    return contract_with_J_times_C(symmetric_gradient_increment, cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const
{
  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function cannot be called with `cache_level` < 2."));

    return (std::numeric_limits<Number>::quiet_NaN() *
            get_identity_symmetric_tensor<dim, Number>());
  }
  else
  {
    symmetric_tensor const C     = C_coefficients.get_coefficient_cell(cell, q);
    scalar const           J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    scalar const           c1    = c1_coefficients.get_coefficient_cell(cell, q);
    scalar const           c2    = c2_coefficients.get_coefficient_cell(cell, q);

    symmetric_tensor result = symmetric_gradient_increment * (-2.0 * c1);
    result +=
      ((-4.0 * ONE_THIRD) * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) * C;
    add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

    return result;
  }
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
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  }

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
