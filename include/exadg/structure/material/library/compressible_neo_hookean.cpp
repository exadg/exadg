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
#include <exadg/structure/material/library/compressible_neo_hookean.h>
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
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  CompressibleNeoHookean(dealii::MatrixFree<dim, Number> const & matrix_free,
                         unsigned int const                      dof_index,
                         unsigned int const                      quad_index,
                         CompressibleNeoHookeanData<dim> const & data,
                         bool const                              spatial_integration,
                         bool const                              force_material_residual)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    parameters_are_variable(data.shear_modulus_function != nullptr or
                            data.lambda_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual)
{
  // initialize (potentially variable) parameters
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored       = dealii::make_vectorized_array<Number>(shear_modulus);

  Number const lambda = data.lambda;
  lambda_stored       = dealii::make_vectorized_array<Number>(lambda);

  if(parameters_are_variable)
  {
    // allocate vectors for variable coefficients and initialize with constant values
    shear_modulus_coefficients.initialize(matrix_free, quad_index, false, false);
    shear_modulus_coefficients.set_coefficients(shear_modulus);

    lambda_coefficients.initialize(matrix_free, quad_index, false, false);
    lambda_coefficients.set_coefficients(lambda);

    VectorType dummy;
    matrix_free.cell_loop(
      &CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
        cell_loop_set_coefficients,
      this,
      dummy,
      dummy);
  }

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement vector.
  if constexpr(cache_level > 0)
  {
    log_J_coefficients.initialize(matrix_free, quad_index, false, false);
    log_J_coefficients.set_coefficients(0.0);

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
        kirchhoff_stress_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

        if(force_material_residual)
        {
          gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
          gradient_displacement_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

          second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                                quad_index,
                                                                false,
                                                                false);
          second_piola_kirchhoff_stress_coefficients.set_coefficients(
            get_zero_tensor<dim, Number>());
        }
      }
      else
      {
        gradient_displacement_coefficients.initialize(matrix_free, quad_index, false, false);
        gradient_displacement_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

        second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                              quad_index,
                                                              false,
                                                              false);
        second_piola_kirchhoff_stress_coefficients.set_coefficients(get_zero_tensor<dim, Number>());

        F_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        F_inv_coefficients.set_coefficients(get_identity_tensor<dim, Number>());

        C_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        C_inv_coefficients.set_coefficients(get_identity_tensor<dim, Number>());
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
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
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

      dealii::VectorizedArray<Number> lambda_vec =
        FunctionEvaluator<0, dim, Number>::value(*(data.lambda_function),
                                                 integrator.quadrature_point(q),
                                                 0.0 /*time*/);

      // set the coefficients
      shear_modulus_coefficients.set_coefficient_cell(cell, q, shear_modulus_vec);

      lambda_coefficients.set_coefficient_cell(cell, q, lambda_vec);
    }
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
void
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const
{
  AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));

  for(unsigned int q = 0; q < integrator_lin->n_q_points; ++q)
  {
    tensor Grad_d_lin = integrator_lin->get_gradient(q);
    auto [F, Jm1] = compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(Grad_d_lin);

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

    scalar const log_J = get_log_J<true /* force_evaluation */>(Jm1, cell, q);
    log_J_coefficients.set_coefficient_cell(cell, q, log_J);

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / (Jm1 + 1.0));
    }

    if constexpr(cache_level > 1)
    {
      if(spatial_integration)
      {
        if constexpr(stable_formulation)
        {
          tensor const tau =
            compute_tau_stable(Grad_d_lin, log_J, shear_modulus_stored, lambda_stored);
          kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);
        }
        else
        {
          tensor const tau = compute_tau_unstable(F, log_J, shear_modulus_stored, lambda_stored);
          kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);
        }

        if(force_material_residual)
        {
          gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);

          tensor const F_inv = invert(F);
          tensor const C_inv = F_inv * transpose(F_inv);
          if constexpr(stable_formulation)
          {
            tensor const S =
              compute_S_stable(Grad_d_lin, C_inv, log_J, shear_modulus_stored, lambda_stored);
            second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
          }
          else
          {
            tensor const S = compute_S_unstable(C_inv, log_J, shear_modulus_stored, lambda_stored);
            second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
          }
        }
      }
      else
      {
        gradient_displacement_coefficients.set_coefficient_cell(cell, q, Grad_d_lin);

        tensor const F_inv = invert(F);
        F_inv_coefficients.set_coefficient_cell(cell, q, F_inv);

        tensor const C_inv = F_inv * transpose(F_inv);
        C_inv_coefficients.set_coefficient_cell(cell, q, C_inv);

        if constexpr(stable_formulation)
        {
          tensor const S =
            compute_S_stable(Grad_d_lin, C_inv, log_J, shear_modulus_stored, lambda_stored);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
        }
        else
        {
          tensor const S = compute_S_unstable(C_inv, log_J, shear_modulus_stored, lambda_stored);
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
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::get_log_J(
  scalar const &     Jm1,
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level == 0 or force_evaluation)
  {
    if constexpr(stable_formulation)
    {
      return log1p(Jm1);
    }
    else
    {
      return log(Jm1 + 1.0);
    }
  }
  else
  {
    return log_J_coefficients.get_coefficient_cell(cell, q);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    if(parameters_are_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
      lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
    }

    if constexpr(cache_level == 0)
    {
      auto const [F, Jm1] =
        compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
      scalar const log_J = get_log_J<false /* force_evaluation*/>(Jm1, cell, q);
      tensor const F_inv = invert(F);
      tensor const C_inv = F_inv * transpose(F_inv);

      if constexpr(stable_formulation)
      {
        return compute_S_stable(
          gradient_displacement, C_inv, log_J, shear_modulus_stored, lambda_stored);
      }
      else
      {
        return compute_S_unstable(C_inv, log_J, shear_modulus_stored, lambda_stored);
      }
    }
    else
    {
      tensor const F =
        compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
      scalar const log_J = log_J_coefficients.get_coefficient_cell(cell, q);
      tensor const F_inv = invert(F);
      tensor const C_inv = F_inv * transpose(F_inv);

      if constexpr(stable_formulation)
      {
        return compute_S_stable(
          gradient_displacement, C_inv, log_J, shear_modulus_stored, lambda_stored);
      }
      else
      {
        return compute_S_unstable(C_inv, log_J, shear_modulus_stored, lambda_stored);
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
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                   "this `cache_level` does not store tensorial quantities."));
    return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
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
inline dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::compute_S_stable(
  tensor const & gradient_displacement,
  tensor const & C_inv,
  scalar const & log_J,
  scalar const & shear_modulus,
  scalar const & lambda) const
{
  tensor S = compute_E_scaled<dim, Number, scalar, stable_formulation>(gradient_displacement,
                                                                       2.0 * shear_modulus);

  add_scaled_identity(S, 2.0 * lambda * log_J);

  return (C_inv * S);
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_S_unstable(tensor const & C_inv,
                     scalar const & log_J,
                     scalar const & shear_modulus,
                     scalar const & lambda) const
{
  tensor S = (2.0 * lambda * log_J - shear_modulus) * C_inv;

  add_scaled_identity(S, shear_modulus);

  return S;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const
{
  if(parameters_are_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
  }

  // Access the stored coefficients precomputed using the last linearization vector.
  if constexpr(cache_level == 0)
  {
    auto const [F, Jm1] =
      compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const log_J = get_log_J<false /*force_evaluation*/>(Jm1, cell, q);
    tensor const F_inv = invert(F);
    tensor const C_inv = F_inv * transpose(F_inv);

    tensor const F_inv_times_gradient_increment = F_inv * gradient_increment;
    scalar const one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
    tensor const Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
    tensor const Dd_C_inv =
      Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

    return ((2.0 * lambda_stored * log_J - shear_modulus_stored) * Dd_C_inv +
            (2.0 * lambda_stored * one_over_J_times_Dd_J) * C_inv);
  }
  else if constexpr(cache_level == 1)
  {
    tensor const F =
      compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const log_J = log_J_coefficients.get_coefficient_cell(cell, q);
    tensor const F_inv = invert(F);
    tensor const C_inv = F_inv * transpose(F_inv);

    tensor const F_inv_times_gradient_increment = F_inv * gradient_increment;
    scalar const one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
    tensor const Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
    tensor const Dd_C_inv =
      Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

    return ((2.0 * lambda_stored * log_J - shear_modulus_stored) * Dd_C_inv +
            (2.0 * lambda_stored * one_over_J_times_Dd_J) * C_inv);
  }
  else
  {
    scalar const log_J = log_J_coefficients.get_coefficient_cell(cell, q);
    tensor const F_inv = F_inv_coefficients.get_coefficient_cell(cell, q);
    tensor const C_inv = C_inv_coefficients.get_coefficient_cell(cell, q);

    tensor const F_inv_times_gradient_increment = F_inv * gradient_increment;
    scalar const one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
    tensor const Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
    tensor const Dd_C_inv =
      Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

    return ((2.0 * lambda_stored * log_J - shear_modulus_stored) * Dd_C_inv +
            (2.0 * lambda_stored * one_over_J_times_Dd_J) * C_inv);
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    if(parameters_are_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
      lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
    }

    if constexpr(cache_level == 0)
    {
      if constexpr(stable_formulation)
      {
        scalar const Jm1 =
          compute_modified_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
        scalar const log_J = get_log_J<false /* force_evaluation */>(Jm1, cell, q);
        return compute_tau_stable(gradient_displacement,
                                  log_J,
                                  shear_modulus_stored,
                                  lambda_stored);
      }
      else
      {
        auto const [F, Jm1] = compute_modified_F_Jm1<dim, Number, check_type, stable_formulation>(
          gradient_displacement);
        scalar const log_J = get_log_J<false /* force_evaluation */>(Jm1, cell, q);
        return compute_tau_unstable(F, log_J, shear_modulus_stored, lambda_stored);
      }
    }
    else
    {
      scalar const log_J = log_J_coefficients.get_coefficient_cell(cell, q);

      if constexpr(stable_formulation)
      {
        return compute_tau_stable(gradient_displacement,
                                  log_J,
                                  shear_modulus_stored,
                                  lambda_stored);
      }
      else
      {
        tensor const F =
          compute_modified_F<dim, Number, check_type, stable_formulation>(gradient_displacement);
        return compute_tau_unstable(F, log_J, shear_modulus_stored, lambda_stored);
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
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::kirchhoff_stress(
  unsigned int const cell,
  unsigned int const q) const
{
  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                   "this `cache_level` does not store tensorial quantities."));
    return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
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
inline dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_tau_stable(tensor const & gradient_displacement,
                     scalar const & log_J,
                     scalar const & shear_modulus,
                     scalar const & lambda) const
{
  tensor tau = shear_modulus * (gradient_displacement + transpose(gradient_displacement) +
                                gradient_displacement * transpose(gradient_displacement));

  add_scaled_identity(tau, 2.0 * lambda * log_J);

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
inline dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  compute_tau_unstable(tensor const & F,
                       scalar const & log_J,
                       scalar const & shear_modulus,
                       scalar const & lambda) const
{
  tensor tau = shear_modulus * (F * transpose(F));

  add_scaled_identity(tau, 2.0 * lambda * log_J - shear_modulus);

  return tau;
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(tensor const &     symmetric_gradient_increment,
                          tensor const &     gradient_displacement,
                          unsigned int const cell,
                          unsigned int const q) const
{
  if(parameters_are_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level == 0)
  {
    scalar const Jm1 =
      compute_modified_Jm1<dim, Number, check_type, stable_formulation>(gradient_displacement);
    scalar const log_J = get_log_J<false /* force_evaluation */>(Jm1, cell, q);
    tensor       result =
      (2.0 * (shear_modulus_stored - 2.0 * lambda_stored * log_J)) * symmetric_gradient_increment;
    add_scaled_identity(result, 2.0 * lambda_stored * trace(symmetric_gradient_increment));

    return result;
  }
  else if constexpr(cache_level == 1)
  {
    // Note that this choice here is only due to compatibility reasons.
    // In general, we want to differentiate between `cache_level` in [0,1] and 2.
    scalar const log_J = log_J_coefficients.get_coefficient_cell(cell, q);
    tensor       result =
      (2.0 * (shear_modulus_stored - 2.0 * lambda_stored * log_J)) * symmetric_gradient_increment;
    add_scaled_identity(result, 2.0 * lambda_stored * trace(symmetric_gradient_increment));

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
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  contract_with_J_times_C(tensor const &     symmetric_gradient_increment,
                          unsigned int const cell,
                          unsigned int const q) const
{
  if(parameters_are_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
  }

  if constexpr(cache_level < 2)
  {
    AssertThrow(cache_level > 1,
                dealii::ExcMessage("This function cannot be called with `cache_level` < 2."));
    tensor result;
    return result;
  }
  else
  {
    // Note that is is a special case here that `cache_level` 1 and 2 coincide,
    // but in the nonlinear operator `cache_level` 1 requires computing the
    // gradient_displacement anyways, such that we use the above function.
    scalar const log_J = log_J_coefficients.get_coefficient_cell(cell, q);
    tensor       result =
      (2.0 * (shear_modulus_stored - 2.0 * lambda_stored * log_J)) * symmetric_gradient_increment;
    add_scaled_identity(result, 2.0 * lambda_stored * trace(symmetric_gradient_increment));

    return result;
  }
}

template<int dim,
         typename Number,
         unsigned int check_type,
         bool         stable_formulation,
         unsigned int cache_level>
dealii::VectorizedArray<Number>
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::one_over_J(
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
CompressibleNeoHookean<dim, Number, check_type, stable_formulation, cache_level>::
  gradient_displacement(unsigned int const cell, unsigned int const q) const
{
  AssertThrow(cache_level > 1,
              dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  return (gradient_displacement_coefficients.get_coefficient_cell(cell, q));
}

// clang-format off
// Note that the higher check types (third template argument) are missing.
template class CompressibleNeoHookean<2, float,  0, true,  0>;
template class CompressibleNeoHookean<2, float,  0, true,  1>;
template class CompressibleNeoHookean<2, float,  0, true,  2>;

template class CompressibleNeoHookean<2, float,  0, false, 0>;
template class CompressibleNeoHookean<2, float,  0, false, 1>;
template class CompressibleNeoHookean<2, float,  0, false, 2>;

template class CompressibleNeoHookean<2, double, 0, true,  0>;
template class CompressibleNeoHookean<2, double, 0, true,  1>;
template class CompressibleNeoHookean<2, double, 0, true,  2>;

template class CompressibleNeoHookean<2, double, 0, false, 0>;
template class CompressibleNeoHookean<2, double, 0, false, 1>;
template class CompressibleNeoHookean<2, double, 0, false, 2>;

template class CompressibleNeoHookean<3, float,  0, true,  0>;
template class CompressibleNeoHookean<3, float,  0, true,  1>;
template class CompressibleNeoHookean<3, float,  0, true,  2>;

template class CompressibleNeoHookean<3, float,  0, false, 0>;
template class CompressibleNeoHookean<3, float,  0, false, 1>;
template class CompressibleNeoHookean<3, float,  0, false, 2>;

template class CompressibleNeoHookean<3, double, 0, true,  0>;
template class CompressibleNeoHookean<3, double, 0, true,  1>;
template class CompressibleNeoHookean<3, double, 0, true,  2>;

template class CompressibleNeoHookean<3, double, 0, false, 0>;
template class CompressibleNeoHookean<3, double, 0, false, 1>;
template class CompressibleNeoHookean<3, double, 0, false, 2>;
// clang-format on

} // namespace Structure
} // namespace ExaDG
