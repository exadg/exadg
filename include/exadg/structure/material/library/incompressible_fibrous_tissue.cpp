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

// deal.II
#include <deal.II/base/vectorization.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/incompressible_fibrous_tissue.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
IncompressibleFibrousTissue<dim, Number>::IncompressibleFibrousTissue(
  dealii::MatrixFree<dim, Number> const &      matrix_free,
  unsigned int const                           dof_index,
  unsigned int const                           quad_index,
  IncompressibleFibrousTissueData<dim> const & data,
  bool const                                   spatial_integration,
  bool const                                   force_material_residual,
  unsigned int const                           check_type,
  unsigned int const                           cache_level)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    shear_modulus_is_variable(data.shear_modulus_function != nullptr),
    spatial_integration(spatial_integration),
    force_material_residual(force_material_residual),
    check_type(check_type),
    cache_level(cache_level)
{
  // initialize (potentially variable) shear modulus
  Number const shear_modulus = data.shear_modulus;
  shear_modulus_stored       = dealii::make_vectorized_array<Number>(shear_modulus);

  bulk_modulus = static_cast<Number>(data.bulk_modulus);

  if(shear_modulus_is_variable)
  {
    // Allocate vectors for variable coefficients and initialize with constant values.
    shear_modulus_coefficients.initialize(matrix_free, quad_index, false, false);
    shear_modulus_coefficients.set_coefficients(shear_modulus);
  }

  // Store variable shear modulus and fiber orientation data.
  fiber_sin_phi.resize(n_fiber_families);
  fiber_cos_phi.resize(n_fiber_families);
  for(unsigned int i = 0; i < n_fiber_families; i++)
  {
    AssertThrow(
      n_fiber_families <= 2,
      dealii::ExcMessage(
        "Assuming potentially symmetrically dispersed fiber families. The fiber "
        "families >2 have the same mean fiber directions as fiber families 1 and 2."));
    fiber_sin_phi[i] =
      std::sin(pow(-1.0, i + 1) * data.fiber_angle_phi_in_degree * dealii::numbers::PI / 180.0);
    fiber_cos_phi[i] =
      std::cos(pow(-1.0, i + 1) * data.fiber_angle_phi_in_degree * dealii::numbers::PI / 180.0);
  }

  fiber_H_11 = data.fiber_H_11;
  fiber_H_22 = data.fiber_H_22;
  fiber_H_33 = data.fiber_H_33;
  fiber_k_1  = data.fiber_k_1;
  fiber_k_2  = data.fiber_k_2;

  VectorType dummy;
  matrix_free.cell_loop(&IncompressibleFibrousTissue<dim, Number>::cell_loop_set_coefficients,
                        this,
                        dummy,
                        dummy);

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement vector.
  if(cache_level > 0)
  {
    J_pow_coefficients.initialize(matrix_free, quad_index, false, false);
    J_pow_coefficients.set_coefficients(1.0);

    c_1_coefficients.initialize(matrix_free, quad_index, false, false);
    c_1_coefficients.set_coefficients(-shear_modulus * one_third * static_cast<Number>(dim));
    c_2_coefficients.initialize(matrix_free, quad_index, false, false);
    c_2_coefficients.set_coefficients(
      shear_modulus * one_third * 2.0 * one_third * static_cast<Number>(dim) + bulk_modulus);

    // Set scalar linearization data for fiber contribution initially. ##++
    fiber_switch_coefficients.resize(n_fiber_families);
    E_i_coefficients.resize(n_fiber_families);
    c_3_coefficients.resize(n_fiber_families);
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      fiber_switch_coefficients[i].initialize(matrix_free, quad_index, false, false);
      fiber_switch_coefficients[i].set_coefficients(1.0);
      E_i_coefficients[i].initialize(matrix_free, quad_index, false, false);
      E_i_coefficients[i].set_coefficients(0.0);
      c_3_coefficients[i].initialize(matrix_free, quad_index, false, false);
      c_3_coefficients[i].set_coefficients(2.0 * fiber_k_1);
    }

    if(spatial_integration)
    {
      one_over_J_coefficients.initialize(matrix_free, quad_index, false, false);
      one_over_J_coefficients.set_coefficients(1.0);
    }

    if(cache_level > 1)
    {
      if(spatial_integration)
      {
        kirchhoff_stress_coefficients.initialize(matrix_free, quad_index, false, false);
        kirchhoff_stress_coefficients.set_coefficients(tensor_zero);

        C_coefficients.initialize(matrix_free, quad_index, false, false);
        C_coefficients.set_coefficients(I);

        // Set tensorial linearization data for fiber contribution initially. ##++
        H_i_times_C_coefficients.resize(n_fiber_families);
        C_times_H_i_coefficients.resize(n_fiber_families);
        for(unsigned int i = 0; i < n_fiber_families; i++)
        {
          H_i_times_C_coefficients[i].initialize(matrix_free, quad_index, false, false);
          H_i_times_C_coefficients[i].set_coefficients(tensor_zero);
          C_times_H_i_coefficients[i].initialize(matrix_free, quad_index, false, false);
          C_times_H_i_coefficients[i].set_coefficients(tensor_zero);
        }

        if(force_material_residual)
        {
          deformation_gradient_coefficients.initialize(matrix_free, quad_index, false, false);
          deformation_gradient_coefficients.set_coefficients(I);

          second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                                quad_index,
                                                                false,
                                                                false);
          second_piola_kirchhoff_stress_coefficients.set_coefficients(tensor_zero);
        }
      }
      else
      {
        deformation_gradient_coefficients.initialize(matrix_free, quad_index, false, false);
        deformation_gradient_coefficients.set_coefficients(I);

        second_piola_kirchhoff_stress_coefficients.initialize(matrix_free,
                                                              quad_index,
                                                              false,
                                                              false);
        second_piola_kirchhoff_stress_coefficients.set_coefficients(tensor_zero);

        F_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        F_inv_coefficients.set_coefficients(I);

        C_inv_coefficients.initialize(matrix_free, quad_index, false, false);
        C_inv_coefficients.set_coefficients(I);
      }

      AssertThrow(cache_level < 3, dealii::ExcMessage("Cache level > 2 not implemented."));
    }
  }
}

template<int dim, typename Number>
void
IncompressibleFibrousTissue<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const &,
  Range const & cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);

  // The material coordinate system is kept constant in the entire domain,
  // where one might consider custom-made material coordinate systems in the future.
  // Phi is the angle from the circumferential vector E_1 towards the longitudinal
  // vector E_2.
  dealii::Tensor<1, dim> E_1, E_2;
  E_1[0] = 1.0;
  E_2[1] = 1.0;

  std::vector<vector> M_1(2), M_2(2);
  for(unsigned int i = 0; i < 2; i++)
  {
    M_1[i] = fiber_cos_phi[i] * E_1 - fiber_sin_phi[i] * E_2;
    M_2[i] = fiber_sin_phi[i] * E_1 + fiber_cos_phi[i] * E_2;
  }

  dealii::Tensor<1, dim> E_3 = cross_product_3d(E_1, E_2);
  vector const M_3 = E_3;

  // Store only the minimal amount for the general case of having a field of
  // material coordinate systems. The mean fiber direction is always needed, since
  // I_i_star(M_1) is used as a fiber switch.
  fiber_direction_M_1.resize(n_fiber_families);
  for(unsigned int i = 0; i < n_fiber_families; i++)
  {
    fiber_direction_M_1[i].initialize(matrix_free, quad_index, false, false);
  }

  if(cache_level < 2)
  {
    // The structure tensor is reconstructed from M1 and M2 on the fly.
    fiber_direction_M_2.resize(n_fiber_families);
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      fiber_direction_M_2[i].initialize(matrix_free, quad_index, false, false);
    }
  }
  else
  {
    // The structure tensors are stored for each fiber family.
    fiber_structure_tensor.resize(n_fiber_families);
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      fiber_structure_tensor[i].initialize(matrix_free, quad_index, false, false);
    }
  }

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // set shear modulus coefficients.
      if(shear_modulus_is_variable)
      {
        scalar shear_modulus_vec =
          FunctionEvaluator<0, dim, Number>::value(*(data.shear_modulus_function),
                                                   integrator.quadrature_point(q),
                                                   0.0 /*time*/);
        shear_modulus_coefficients.set_coefficient_cell(cell, q, shear_modulus_vec);
      }

      // Fill the fiber orientation vectors or the structure tensor directly
      // based on the Euclidean material coordinate system.
      if(cache_level < 2)
      {
        for(unsigned int i = 0; i < n_fiber_families; i++)
        {
          fiber_direction_M_1[i].set_coefficient_cell(cell, q, M_1[i]);
          fiber_direction_M_2[i].set_coefficient_cell(cell, q, M_2[i]);
        }
      }
      else
      {
        for(unsigned int i = 0; i < n_fiber_families; i++)
        {
          fiber_direction_M_1[i].set_coefficient_cell(cell, q, M_1[i]);

          // clang-format off
    	  fiber_structure_tensor[i].set_coefficient_cell(cell, q, fiber_H_11 * outer_product(M_1[i], M_1[i])
                                                                + fiber_H_22 * outer_product(M_2[i], M_2[i])
                                                                + fiber_H_33 * outer_product(M_3, M_3));
          // clang-format on
        }
      }
    }
  }
}

template<int dim, typename Number>
inline void
IncompressibleFibrousTissue<dim, Number>::get_structure_tensor(tensor &           H,
                                                               vector const &     M_1,
                                                               vector const &     M_2,
                                                               unsigned int const i,
                                                               unsigned int const cell,
                                                               unsigned int const q) const
{
  if(cache_level < 2)
  {
    vector M_3 = cross_product_3d(M_1, M_2);

    // clang format off
    H = fiber_H_11 * outer_product(M_1, M_1) + fiber_H_22 * outer_product(M_2, M_2) +
        fiber_H_33 * outer_product(M_3, M_3);
    // clang format on
  }
  else
  {
    H = fiber_structure_tensor[i].get_coefficient_cell(cell, q);
  }
}

template<int dim, typename Number>
inline void
IncompressibleFibrousTissue<dim, Number>::get_fiber_switch(scalar &           fiber_switch,
                                                           vector const &     M_1,
                                                           tensor const &     C,
                                                           unsigned int const i,
                                                           unsigned int const cell,
                                                           unsigned int const q,
                                                           bool const force_evaluation) const
{
  if(cache_level == 0 or force_evaluation)
  {
    scalar const I_i = scalar_product(outer_product(M_1, M_1), C);

    // fiber_switch = I_i_star < 1 ? 0.0 : 1.0
    fiber_switch = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(I_i,
                                                                                     scalar_one,
                                                                                     scalar_one, // scalar_zero, ##++ fiber switch is always on now
                                                                                     scalar_one);
  }
  else
  {
    fiber_switch = fiber_switch_coefficients[i].get_coefficient_cell(cell, q);
  }
}


template<int dim, typename Number>
void
IncompressibleFibrousTissue<dim, Number>::do_set_cell_linearization_data(
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

    tensor Grad_d_lin = integrator_lin->get_gradient(q);

    scalar J;
    tensor F;
    get_modified_F_J(F, J, Grad_d_lin, check_type, true /* compute_J */);

    // Overwrite computed values with admissible stored ones
    if(check_type == 2)
    {
      tensor const F_old    = deformation_gradient_coefficients.get_coefficient_cell(cell, q);
      bool         update_J = false;
      for(unsigned int i = 0; i < J.size(); ++i)
      {
        if(J[i] <= get_J_tol<Number>())
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
        J = determinant(F);
      }
    }

    scalar const J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
    J_pow_coefficients.set_coefficient_cell(cell, q, J_pow);

    if(spatial_integration)
    {
      one_over_J_coefficients.set_coefficient_cell(cell, q, 1.0 / J);
    }

    tensor const C   = transpose(F) * F;
    scalar const I_1 = trace(C);
    scalar const c1 =
      bulk_modulus * 0.5 * (J * J - 1.0) - shear_modulus_stored * one_third * J_pow * I_1;
    c_1_coefficients.set_coefficient_cell(cell, q, c1);

    scalar const c2 =
      shear_modulus_stored * one_third * one_third * 2.0 * J_pow * I_1 + bulk_modulus * J * J;
    c_2_coefficients.set_coefficient_cell(cell, q, c2);

    // Set scalar linearization data for fiber contribution. ##++
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      vector const M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector       M_2_cache_level_0_1;
      if(cache_level < 2)
      {
        M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      }

      scalar fiber_switch;
      get_fiber_switch(fiber_switch, M_1, C, i, cell, q, true /* force_evaluation */);
      fiber_switch_coefficients[i].set_coefficient_cell(cell, q, fiber_switch);

      // Compute or load structure tensor.
      tensor H_i;
      get_structure_tensor(H_i, M_1, M_2_cache_level_0_1, i, cell, q);

      tensor const C_minus_I = C - I;
      scalar       E_i       = scalar_product(H_i, C_minus_I);
      E_i_coefficients[i].set_coefficient_cell(cell, q, E_i);

      scalar c3 = 2.0 * fiber_k_1 * exp(fiber_k_2 * E_i * E_i);
      c_3_coefficients[i].set_coefficient_cell(cell, q, c3);
    }

    if(cache_level > 1)
    {
      if(spatial_integration)
      {
        tensor const tau_lin =
          this->kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau_lin);

        C_coefficients.set_coefficient_cell(cell, q, C);

        // Set tensorial linearization data for fiber contribution. ##++
        for(unsigned int i = 0; i < n_fiber_families; i++)
        {
          tensor const H_i = fiber_structure_tensor[i].get_coefficient_cell(cell, q);
          H_i_times_C_coefficients[i].set_coefficient_cell(cell, q, H_i * C);
          C_times_H_i_coefficients[i].set_coefficient_cell(cell, q, C * H_i);
        }

        if(force_material_residual)
        {
          deformation_gradient_coefficients.set_coefficient_cell(cell, q, F);
          tensor const S_lin =
            this->second_piola_kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
          second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S_lin);
        }
      }
      else
      {
        deformation_gradient_coefficients.set_coefficient_cell(cell, q, F);

        tensor const S_lin =
          this->second_piola_kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        second_piola_kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, S_lin);

        tensor const F_inv = invert(F);
        F_inv_coefficients.set_coefficient_cell(cell, q, F_inv);

        tensor const C_inv = F_inv * transpose(F_inv);
        C_inv_coefficients.set_coefficient_cell(cell, q, C_inv);
      }
    }
  }
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number>::second_piola_kirchhoff_stress(
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
      c1 = bulk_modulus * 0.5 * (J * J - 1.0) - shear_modulus_stored * J_pow * one_third * trace(C);
    }
    else
    {
      J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      c1    = c_1_coefficients.get_coefficient_cell(cell, q);
    }

    S = invert(C) * c1;
    add_scaled_identity(S, shear_modulus_stored * J_pow);

    // Add fiber contribution.
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      // Compute or load fiber switch. ##++ S
      vector M_1_cache_level_0_1, M_2_cache_level_0_1;
      if(cache_level < 2)
      {
        M_1_cache_level_0_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      }
      else
      {
        // M_1 : dummy vector suffices for function call.
        // M_2 : dummy vector suffices for function call.
      }
      scalar fiber_switch;
      get_fiber_switch(
        fiber_switch, M_1_cache_level_0_1, C, i, cell, q, false /* force_evaluation */);

      // Compute or load structure tensor.
      tensor H_i;
      get_structure_tensor(H_i, M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

      // Compute or load fiber strain-like quantity.
      scalar E_i;
      if(cache_level == 0)
      {
        tensor C_minus_I = C - I;
        E_i              = scalar_product(H_i, C_minus_I);
      }
      else
      {
        E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      }

      // Compute or load c3 coefficient.
      scalar c3;
      if(cache_level == 0)
      {
        c3 = 2.0 * fiber_k_1 * exp(fiber_k_2 * E_i * E_i);
      }
      else
      {
        c3 = c_3_coefficients[i].get_coefficient_cell(cell, q);
      }

      S += (fiber_switch * c3 * E_i) * H_i;
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
IncompressibleFibrousTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
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
    scalar const I_1 = trace(transpose(deformation_gradient) * deformation_gradient);

    J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
    c1    = bulk_modulus * 0.5 * (J * J - 1.0) - shear_modulus_stored * one_third * J_pow * I_1;
    c2    = shear_modulus_stored * one_third * J_pow * 2.0 * one_third * I_1 + bulk_modulus * J * J;
  }
  else
  {
    J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
    c1    = c_1_coefficients.get_coefficient_cell(cell, q);
    c2    = c_2_coefficients.get_coefficient_cell(cell, q);
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

  tensor const F_inv_times_gradient_increment = F_inv * gradient_increment;

  scalar const one_over_J_times_Dd_J          = trace(F_inv_times_gradient_increment);
  tensor const Dd_F_inv_times_transpose_F_inv = -F_inv_times_gradient_increment * C_inv;
  tensor const Dd_C_inv =
    Dd_F_inv_times_transpose_F_inv + transpose(Dd_F_inv_times_transpose_F_inv);

  tensor const transpose_gradient_increment_times_F =
    transpose(gradient_increment) * deformation_gradient;

  Dd_S = C_inv * (c2 * one_over_J_times_Dd_J - (2.0 * shear_modulus_stored * one_third * J_pow) *
                                                 trace(transpose_gradient_increment_times_F));
  Dd_S += Dd_C_inv * c1;
  add_scaled_identity(Dd_S,
                      -shear_modulus_stored * one_third * J_pow * 2.0 * one_over_J_times_Dd_J);

  // Add fiber contribution.
  for(unsigned int i = 0; i < n_fiber_families; i++)
  {
    // Compute or load fiber switch. ##++ Dd_S
    vector M_1_cache_level_0_1, M_2_cache_level_0_1;
    tensor C_cache_level_0_1;
    if(cache_level < 2)
    {
      M_1_cache_level_0_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      C_cache_level_0_1   = transpose(deformation_gradient) * deformation_gradient;
    }
    else
    {
      // M_1 : dummy vector suffices for function call.
      // M_2 : dummy vector suffices for function call.
      // C   : dummy tensor suffices for function call.
    }
    scalar fiber_switch;
    get_fiber_switch(fiber_switch,
                     M_1_cache_level_0_1,
                     C_cache_level_0_1,
                     i,
                     cell,
                     q,
                     false /* force_evaluation */);

    // Compute or load structure tensor.
    tensor H_i;
    get_structure_tensor(H_i, M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

    // Compute or load fiber strain-like quantity.
    scalar E_i;
    if(cache_level == 0)
    {
      tensor C_minus_I = C_cache_level_0_1 - I;
      E_i              = scalar_product(H_i, C_minus_I);
    }
    else
    {
      E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
    }

    // Compute or load c3 coefficient.
    scalar c3;
    if(cache_level == 0)
    {
      c3 = 2.0 * fiber_k_1 * exp(fiber_k_2 * E_i * E_i);
    }
    else
    {
      c3 = c_3_coefficients[i].get_coefficient_cell(cell, q);
    }

    Dd_S += H_i * (fiber_switch * c3 * (2.0 * fiber_k_2 * E_i * E_i + 1.0) *
                   scalar_product(H_i,
                                  transpose_gradient_increment_times_F +
                                    transpose(transpose_gradient_increment_times_F)));
  }

  return Dd_S;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                           unsigned int const cell,
                                                           unsigned int const q,
                                                           bool const force_evaluation) const
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

    // Add fiber contribution.
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      // Compute or load fiber switch. ##++ tau
      vector M_1_cache_level_0_1, M_2_cache_level_0_1;
      tensor C_cache_level_0_1;
      if(cache_level < 2)
      {
        M_1_cache_level_0_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
        C_cache_level_0_1   = transpose(F) * F;
      }
      else
      {
        // M_1 : dummy vector suffices for function call.
        // M_2 : dummy vector suffices for function call.
        // C   : dummy tensor suffices for function call.
      }
      scalar fiber_switch;
      get_fiber_switch(fiber_switch,
                       M_1_cache_level_0_1,
                       C_cache_level_0_1,
                       i,
                       cell,
                       q,
                       false /* force_evaluation */);

      // Compute or load structure tensor.
      tensor H_i;
      get_structure_tensor(H_i, M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

      // Compute or load fiber strain-like quantity.
      scalar E_i;
      if(cache_level == 0)
      {
        tensor C_minus_I = C_cache_level_0_1 - I;
        E_i              = scalar_product(H_i, C_minus_I);
      }
      else
      {
        E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      }

      // Compute or load c3 coefficient.
      scalar c3;
      if(cache_level == 0)
      {
        c3 = 2.0 * fiber_k_1 * exp(fiber_k_2 * E_i * E_i);
      }
      else
      {
        c3 = c_3_coefficients[i].get_coefficient_cell(cell, q);
      }

      // Add terms in non-push-forwarded form.
      tau += (c3 * E_i) * H_i;
    }

    scalar J_pow, c1;
    if(cache_level == 0)
    {
      J   = determinant(F);
      tensor const C = transpose(F) * F;

      J_pow = pow(J, static_cast<Number>(-2.0 * one_third));
      c1    = bulk_modulus * 0.5 * (J * J - 1.0) -
           (one_third * shear_modulus_stored * J_pow) * trace(C);
    }
    else
    {
      J_pow = J_pow_coefficients.get_coefficient_cell(cell, q);
      c1    = c_1_coefficients.get_coefficient_cell(cell, q);
    }

    // tau holds fiber terms in non-push-forwarded form.
    // Add isochoric terms (the one with F*transpose(F))
    // to avoid a matrix-matrix product.
    add_scaled_identity(tau, shear_modulus_stored * J_pow);

    // We cannot avoid this triple matrix product even if
    // we have C already computed as for cache_level 0.
    tau = F * tau * transpose(F);

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
IncompressibleFibrousTissue<dim, Number>::contract_with_J_times_C(
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
    c1    = c_1_coefficients.get_coefficient_cell(cell, q);
    c2    = c_2_coefficients.get_coefficient_cell(cell, q);
  }

  result = symmetric_gradient_increment * (-2.0 * c1);
  add_scaled_identity(result,
                      -4.0 * one_third * shear_modulus_stored * J_pow *
                          scalar_product(C, symmetric_gradient_increment) +
                        c2 * trace(symmetric_gradient_increment));

  // Add fiber contribution.
  for(unsigned int i = 0; i < n_fiber_families; i++)
  {
    // Compute or load fiber switch. ##++ J C : (o)
    vector M_1_cache_level_0_1, M_2_cache_level_0_1;
    if(cache_level < 2)
    {
      M_1_cache_level_0_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
    }
    else
    {
      // M_1 : dummy vector suffices for function call.
      // M_2 : dummy vector suffices for function call.
      // C   : dummy tensor suffices for function call.
    }
    scalar fiber_switch;
    get_fiber_switch(fiber_switch,
                     M_1_cache_level_0_1,
                     C,
                     i,
                     cell,
                     q,
                     false /* force_evaluation */);

    // Compute or load structure tensor.
    tensor H_i;
    get_structure_tensor(H_i, M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

    // Compute or load fiber strain-like quantity.
    scalar E_i;
    if(cache_level == 0)
    {
      tensor C_minus_I = C - I;
      E_i              = scalar_product(H_i, C_minus_I);
    }
    else
    {
      E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
    }

    // Compute or load c3 coefficient.
    scalar c3;
    if(cache_level == 0)
    {
      c3 = 2.0 * fiber_k_1 * exp(fiber_k_2 * E_i * E_i);
    }
    else
    {
      c3 = c_3_coefficients[i].get_coefficient_cell(cell, q);
    }

    // Compute or load H_i * C tensor.
    tensor H_i_times_C, C_times_H_i;
    if(cache_level < 2)
    {
      H_i_times_C = H_i * C;
      C_times_H_i = C * H_i;
    }
    else
    {
      H_i_times_C = H_i_times_C_coefficients[i].get_coefficient_cell(cell, q);
      C_times_H_i = C_times_H_i_coefficients[i].get_coefficient_cell(cell, q);
    }

    result += (4.0 * c3 * (2.0 * fiber_k_2 * E_i * E_i + 1.0) * scalar_product(C_times_H_i, symmetric_gradient_increment) ) * H_i_times_C;
  }

  return result;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number>::one_over_J(unsigned int const cell,
                                                     unsigned int const q) const
{
  AssertThrow(spatial_integration and cache_level > 0,
              dealii::ExcMessage("Cannot access precomputed one_over_J."));
  return (one_over_J_coefficients.get_coefficient_cell(cell, q));
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
IncompressibleFibrousTissue<dim, Number>::deformation_gradient(unsigned int const cell,
                                                               unsigned int const q) const
{
  AssertThrow(cache_level > 1,
              dealii::ExcMessage("Cannot access precomputed deformation gradient."));
  return (deformation_gradient_coefficients.get_coefficient_cell(cell, q));
}

template class IncompressibleFibrousTissue<2, float>;
template class IncompressibleFibrousTissue<2, double>;

template class IncompressibleFibrousTissue<3, float>;
template class IncompressibleFibrousTissue<3, double>;

} // namespace Structure
} // namespace ExaDG
