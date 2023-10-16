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
template<int dim, typename Number>
CompressibleNeoHookean<dim, Number>::CompressibleNeoHookean(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  CompressibleNeoHookeanData<dim> const & data,
  bool const                              spatial_integration,
  bool const                              force_material_residual,
  unsigned int const                      check_type,
  unsigned int const                      cache_level)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    parameters_are_variable(data.shear_modulus_function != nullptr or
                            data.lambda_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual),
	check_type(check_type),
    cache_level(cache_level)
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
    matrix_free.cell_loop(&CompressibleNeoHookean<dim, Number>::cell_loop_set_coefficients,
                          this,
                          dummy,
                          dummy);
  }

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement vector.
  if(cache_level > 0)
  {
    log_J_coefficients.initialize(matrix_free, quad_index, false, false);
    log_J_coefficients.set_coefficients(0.0);

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
CompressibleNeoHookean<dim, Number>::do_set_cell_linearization_data(
  std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
  unsigned int const                                                         cell) const
{
  AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));

  for(unsigned int q = 0; q < integrator_lin->n_q_points; ++q)
  {
	tensor const Grad_d_lin = integrator_lin->get_gradient(q);

    scalar J;
    tensor F;
    get_modified_F_J(F, J, Grad_d_lin, check_type, true /* compute_J */);

    log_J_coefficients.set_coefficient_cell(cell, q, log(J));

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / J);
    }

    if(cache_level > 1)
    {
      deformation_gradient_coefficients.set_coefficient_cell(cell, q, F);

      if(spatial_integration)
      {
        tensor const tau_lin =
          this->kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau_lin);
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
CompressibleNeoHookean<dim, Number>::cell_loop_set_coefficients(
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

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q,
  bool const         force_evaluation) const
{
  tensor S;
  if(cache_level < 2 or force_evaluation)
  {
    if(parameters_are_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
      lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
    }

    scalar J;
    tensor F;
    get_modified_F_J(F, J, gradient_displacement, check_type, false /* compute_J */);

    // Access the stored coefficients precomputed using the last linearization vector.
    scalar log_J;
    if(cache_level == 0)
    {
      J     = determinant(F);
      log_J = log(J);
    }
    else
    {
      log_J = log_J_coefficients.get_coefficient_cell(cell, q);
    }

    tensor const F_inv = invert(F);
    tensor const C_inv = F_inv * transpose(F_inv);

    S = (2.0 * lambda_stored * log_J - shear_modulus_stored) * C_inv;
    add_scaled_identity(S, shear_modulus_stored);
  }
  else
  {
    S = second_piola_kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }

  return S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  if(parameters_are_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
  }

  // Access the stored coefficients precomputed using the last linearization vector.
  scalar log_J;
  if(cache_level == 0)
  {
    log_J = log(determinant(deformation_gradient));
  }
  else
  {
    log_J = log_J_coefficients.get_coefficient_cell(cell, q);
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

  scalar const one_over_J_times_Dd_J          = trace(F_inv * gradient_increment);
  tensor const Dd_F_inv_times_transpose_F_inv = -F_inv * gradient_increment * C_inv;
  tensor const Dd_C_inv =
    Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

  return (-(shear_modulus_stored - 2.0 * lambda_stored * log_J) * Dd_C_inv +
          (2.0 * lambda_stored * one_over_J_times_Dd_J) * C_inv);
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                      unsigned int const cell,
                                                      unsigned int const q,
                                                      bool const         force_evaluation) const
{
  tensor tau;
  if(cache_level < 2 or force_evaluation)
  {
    if(parameters_are_variable)
    {
      shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
      lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
    }

    scalar J;
    tensor F;
    get_modified_F_J(F, J, gradient_displacement, check_type, false /* compute_J */);

    // Access the stored coefficients precomputed using the last linearization vector.
    scalar log_J;
    if(cache_level == 0)
    {
      J     = determinant(F);
      log_J = log(J);
    }
    else
    {
      log_J = log_J_coefficients.get_coefficient_cell(cell, q);
    }

    tau = shear_modulus_stored * (F * transpose(F));
    add_scaled_identity(tau, 2.0 * lambda_stored * log_J - shear_modulus_stored);
  }
  else
  {
    tau = kirchhoff_stress_coefficients.get_coefficient_cell(cell, q);
  }

  return tau;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number>::contract_with_J_times_C(
  tensor const &     symmetric_gradient_increment,
  tensor const &     deformation_gradient,
  unsigned int const cell,
  unsigned int const q) const
{
  if(parameters_are_variable)
  {
    shear_modulus_stored = shear_modulus_coefficients.get_coefficient_cell(cell, q);
    lambda_stored        = lambda_coefficients.get_coefficient_cell(cell, q);
  }

  // Access the stored coefficients precomputed using the last linearization vector.
  scalar log_J;
  if(cache_level == 0)
  {
    log_J = log(determinant(deformation_gradient));
  }
  else
  {
    log_J = log_J_coefficients.get_coefficient_cell(cell, q);
  }

  tensor result =
    (2.0 * (shear_modulus_stored - 2.0 * lambda_stored * log_J)) * symmetric_gradient_increment;
  add_scaled_identity(result, (2.0 * lambda_stored * trace(symmetric_gradient_increment)));

  return result;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
CompressibleNeoHookean<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
{
  AssertThrow(spatial_integration and cache_level > 0,
              dealii::ExcMessage("Cannot access precomputed one_over_J."));
  return (one_over_J_coefficients.get_coefficient_cell(cell, q));
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
CompressibleNeoHookean<dim, Number>::deformation_gradient(unsigned int const cell,
                                                          unsigned int const q) const
{
  AssertThrow(cache_level > 1,
              dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  return (deformation_gradient_coefficients.get_coefficient_cell(cell, q));
}

template class CompressibleNeoHookean<2, float>;
template class CompressibleNeoHookean<2, double>;

template class CompressibleNeoHookean<3, float>;
template class CompressibleNeoHookean<3, double>;

} // namespace Structure
} // namespace ExaDG
