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
  unsigned int const                        check_type,
  bool const                                stable_formulation,
  unsigned int const                        cache_level)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual),
    stable_formulation(stable_formulation),
    check_type(check_type),
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

    if(cache_level > 1)
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
          deformation_gradient_coefficients.initialize(matrix_free, quad_index, false, false);
          deformation_gradient_coefficients.set_coefficients(get_identity<dim, Number>());

          second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                                quad_index,
                                                                false,
                                                                false);
          second_piola_kirchhoff_stress_coefficients.set_coefficients(zero_tensor);
        }
      }
      else
      {
        deformation_gradient_coefficients.initialize(matrix_free, quad_index, false, false);
        deformation_gradient_coefficients.set_coefficients(get_identity<dim, Number>());

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
void
IncompressibleNeoHookean<dim, Number>::do_set_cell_linearization_data(
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
    get_modified_F_Jm1(F, Jm1, Grad_d_lin, check_type, true /* compute_J */, stable_formulation);

    // Overwrite computed values with admissible stored ones
    if(check_type == 2)
    {
      tensor const F_old    = deformation_gradient_coefficients.get_coefficient_cell(cell, q);
      bool         update_J = false;
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

    scalar const J_pow = get_J_pow(Jm1, true /* force_evaluation */, cell, q);
    J_pow_coefficients.set_coefficient_cell(cell, q, J_pow);

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / (Jm1 + 1.0));
    }

    tensor const E = get_E_scaled<dim, Number, Number>(Grad_d_lin, 1.0, stable_formulation);
    scalar const c1 =
      get_c1(Jm1, J_pow, E, shear_modulus_stored, true /* force_evaluation */, cell, q);
    scalar const c2 =
      get_c2(Jm1, J_pow, E, shear_modulus_stored, true /* force_evaluation */, cell, q);

    c1_coefficients.set_coefficient_cell(cell, q, c1);
    c2_coefficients.set_coefficient_cell(cell, q, c2);

    if(cache_level > 1)
    {
      if(spatial_integration)
      {
        tensor const tau = this->kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau);

        tensor const C = transpose(F) * F;
        C_coefficients.set_coefficient_cell(cell, q, C);

        if(force_material_residual)
        {
          deformation_gradient_coefficients.set_coefficient_cell(cell, q, F);
          tensor const S =
            this->second_piola_kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S);
        }
      }
      else
      {
        deformation_gradient_coefficients.set_coefficient_cell(cell, q, F);

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

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number>::get_c1(scalar const &     Jm1,
                                              scalar const &     J_pow,
                                              tensor const &     E,
                                              scalar const &     shear_modulus_stored,
                                              bool const         force_evaluation,
                                              unsigned int const cell,
                                              unsigned int const q) const
{
  scalar c1;

  if(cache_level == 0 or force_evaluation)
  {
    c1 = 0.5 * bulk_modulus * get_JJm1<Number>(Jm1, stable_formulation) -
         shear_modulus_stored * one_third * J_pow * get_I_1<dim, Number>(E, stable_formulation);
  }
  else
  {
    c1 = c1_coefficients.get_coefficient_cell(cell, q);
  }

  return c1;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number>::get_c2(scalar const &     Jm1,
                                              scalar const &     J_pow,
                                              tensor const &     E,
                                              scalar const &     shear_modulus_stored,
                                              bool const         force_evaluation,
                                              unsigned int const cell,
                                              unsigned int const q) const
{
  scalar c2;

  if(cache_level == 0 or force_evaluation)
  {
    c2 = bulk_modulus * (get_JJm1<Number>(Jm1, stable_formulation) + 1.0) +
         2.0 * shear_modulus_stored * one_third * one_third * J_pow *
           get_I_1<dim, Number>(E, stable_formulation);
  }
  else
  {
    c2 = c2_coefficients.get_coefficient_cell(cell, q);
  }

  return c2;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleNeoHookean<dim, Number>::get_J_pow(scalar const &     Jm1,
                                                 bool const         force_evaluation,
                                                 unsigned int const cell,
                                                 unsigned int const q) const
{
  scalar J_pow;

  if(cache_level == 0 or force_evaluation)
  {
    J_pow = pow(Jm1 + 1.0, static_cast<Number>(-2.0 * one_third));
  }
  else
  {
    J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
  }

  return J_pow;
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

    scalar Jm1;
    tensor F;
    get_modified_F_Jm1(F,
                       Jm1,
                       gradient_displacement,
                       check_type,
                       cache_level == 0 or force_evaluation /* compute_J */,
                       stable_formulation);
    if(cache_level == 1 and not force_evaluation and stable_formulation)
    {
      Jm1 = Jm1_coefficients.get_coefficient_cell(cell, q);
    }

    tensor const C = transpose(F) * F;

    scalar const J_pow = get_J_pow(Jm1, force_evaluation, cell, q);

    if(stable_formulation)
    {
      S = get_E_scaled<dim, Number, scalar>(gradient_displacement,
                                            2.0 * shear_modulus_stored * J_pow,
                                            true /* stable_formulation */);
      add_scaled_identity<dim, Number>(
        S, -one_third * trace(S) + 0.5 * bulk_modulus * get_JJm1<Number>(Jm1, stable_formulation));
      S = invert(C) * S;
    }
    else
    {
      if(cache_level == 0 or force_evaluation)
      {
        S = get_E_scaled<dim, Number, Number>(gradient_displacement,
                                              1.0,
                                              false /* stable_formulation */);
      }
      scalar const c1 =
        get_c1(Jm1, J_pow, S /* E */, shear_modulus_stored, force_evaluation, cell, q);

      S = invert(C) * c1;
      add_scaled_identity(S, shear_modulus_stored * J_pow);
    }
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
  tensor const &     gradient_displacement_cache_level_0_1,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  tensor Dd_S;

  if(shear_modulus_is_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
  }

  scalar Jm1_cache_level_0;
  tensor E_cache_level_0;
  if(cache_level == 0)
  {
    get_modified_F_Jm1(E_cache_level_0 /* F */,
                       Jm1_cache_level_0,
                       gradient_displacement_cache_level_0_1,
                       check_type,
                       true /* compute_J */,
                       stable_formulation);

    E_cache_level_0 = get_E_scaled<dim, Number, Number>(gradient_displacement_cache_level_0_1,
                                                        1.0,
                                                        stable_formulation);
  }
  else
  {
    // Dummy Jm1 and E sufficient.
  }

  scalar const J_pow = get_J_pow(Jm1_cache_level_0, false /* force_evaluation */, cell, q);
  scalar const c1    = get_c1(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           false /* force_evaluation */,
                           cell,
                           q);
  scalar const c2    = get_c2(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           false /* force_evaluation */,
                           cell,
                           q);

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

  tensor const F_inv_times_gradient_increment = F_inv * gradient_increment;

  scalar const one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
  tensor const Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
  tensor const Dd_C_inv =
    Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

  Dd_S = C_inv * (c2 * one_over_J_times_Dd_J -
                  (2.0 * shear_modulus_stored * one_third * J_pow) *
                    trace(transpose(gradient_increment) * deformation_gradient));
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

    scalar Jm1;
    tensor F;
    get_modified_F_Jm1(F,
                       Jm1,
                       gradient_displacement,
                       check_type,
                       cache_level == 0 or force_evaluation /* compute_J */,
                       stable_formulation);
    if(cache_level == 1 and not force_evaluation and stable_formulation)
    {
      Jm1 = Jm1_coefficients.get_coefficient_cell(cell, q);
    }

    scalar const J_pow = get_J_pow(Jm1, force_evaluation, cell, q);

    if(stable_formulation)
    {
      tau = get_E_scaled<dim, Number, scalar>(gradient_displacement,
                                              2.0 * shear_modulus_stored * J_pow,
                                              true /* stable_formulation */);
      add_scaled_identity<dim, Number>(tau,
                                       -one_third * trace(tau) +
                                         0.5 * bulk_modulus *
                                           get_JJm1<Number>(Jm1, stable_formulation));
    }
    else
    {
      if(cache_level == 0 or force_evaluation)
      {
        tau = get_E_scaled<dim, Number, Number>(gradient_displacement, 1.0, stable_formulation);
      }
      else
      {
        // Dummy E sufficient.
      }

      scalar const c1 =
        get_c1(Jm1, J_pow, tau /* E */, shear_modulus_stored, force_evaluation, cell, q);

      tau = (F * transpose(F)) * shear_modulus_stored * J_pow;
      add_scaled_identity(tau, c1);
    }
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
  tensor const &     gradient_displacement_cache_level_0_1,
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

  scalar Jm1_cache_level_0;
  tensor E_cache_level_0;
  if(cache_level == 0)
  {
    get_modified_F_Jm1(result,
                       Jm1_cache_level_0,
                       gradient_displacement_cache_level_0_1,
                       check_type,
                       true /* compute_J */,
                       stable_formulation);

    E_cache_level_0 = get_E_scaled<dim, Number, Number>(gradient_displacement_cache_level_0_1,
                                                        1.0,
                                                        stable_formulation);
  }
  else
  {
    // Dummy E and Jm1 sufficient.
  }

  scalar const J_pow = get_J_pow(Jm1_cache_level_0, false /* force_evaluation */, cell, q);
  scalar const c1    = get_c1(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           false /* force_evaluation */,
                           cell,
                           q);
  scalar const c2    = get_c2(Jm1_cache_level_0,
                           J_pow,
                           E_cache_level_0,
                           shear_modulus_stored,
                           false /* force_evaluation */,
                           cell,
                           q);

  result = symmetric_gradient_increment * (-2.0 * c1);
  result +=
    (-4.0 * one_third * shear_modulus_stored * J_pow * trace(symmetric_gradient_increment)) * C;
  add_scaled_identity(result, c2 * trace(symmetric_gradient_increment));

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
