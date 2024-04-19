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
#include <deal.II/fe/mapping_fe.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/material/library/incompressible_fibrous_tissue.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>

// ExaDG-Bio
#define LINK_TO_EXADGBIO
#ifdef LINK_TO_EXADGBIO
#  include "../../../../../../exadg-bio/include/match_cell_data.h"
#endif

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
  bool const                                   stable_formulation,
  unsigned int const                           cache_level)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    orientation_vectors_provided(data.e1_orientations != nullptr or
                                 data.e1_orientations != nullptr),
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
    // Allocate vectors for variable coefficients and initialize with constant values.
    shear_modulus_coefficients.initialize(matrix_free, quad_index, false, false);
    shear_modulus_coefficients.set_coefficients(shear_modulus);
  }

  // Store variable shear modulus and fiber orientation data.
  fiber_sin_phi.resize(n_fiber_families);
  fiber_cos_phi.resize(n_fiber_families);
  for(unsigned int i = 0; i < n_fiber_families; i++)
  {
    AssertThrow(n_fiber_families <= 2,
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

  // Initialize linearization cache and fill with values corresponding to
  // the initial linearization vector assumed to be a zero displacement
  // vector if possible.
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

    // Set scalar linearization data for fiber contribution initially.
    E_i_coefficients.resize(n_fiber_families);
    c3_coefficients.resize(n_fiber_families);
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      E_i_coefficients[i].initialize(matrix_free, quad_index, false, false);
      E_i_coefficients[i].set_coefficients(0.0);
      c3_coefficients[i].initialize(matrix_free, quad_index, false, false);
      c3_coefficients[i].set_coefficients(2.0 * fiber_k_1);
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

        // Set tensorial linearization data for fiber contribution initially.
        // Note that these have to be corrected in cell_loop_set_coefficients().
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

  // The vectors created or imported from the binary files in the application
  // increase in h-level and then in p-level (ph-Multigrid expected here).
  if(orientation_vectors_provided)
  {
    AssertThrow(data.e1_orientations != nullptr and data.e2_orientations != nullptr,
                dealii::ExcMessage("Provide orientation vectors for both e1 and e2."));

    AssertThrow(data.e1_orientations->size() == data.e2_orientations->size(),
                dealii::ExcMessage("Provide orientation vectors for all levels for e1 and e2."));

    for(unsigned int i = 0; i < data.e1_orientations->size(); i++)
    {
      AssertThrow((*data.e1_orientations)[i].size() == (*data.e2_orientations)[i].size(),
                  dealii::ExcMessage(
                    "Provide e1 and e2 orientation vectors of equal size for each level."));
    }

    AssertThrow(data.e1_orientations->size() == data.degree_per_level.size(),
                dealii::ExcMessage("Provide degree for all levels for e1 and e2."));

    typedef typename IncompressibleFibrousTissueData<dim>::VectorType VectorTypeOrientation;

    e1_orientation = std::make_shared<VectorType>();
    e2_orientation = std::make_shared<VectorType>();
    matrix_free.initialize_dof_vector(*e1_orientation, dof_index);
    matrix_free.initialize_dof_vector(*e2_orientation, dof_index);

#ifdef LINK_TO_EXADGBIO
    // Read the suitable vector from binary format.
    dealii::DoFHandler<dim> const & dof_handler = matrix_free.get_dof_handler(dof_index);
    unsigned int const              degree      = dof_handler.get_fe().base_element(0).degree;
    MPI_Comm const &                mpi_comm    = dof_handler.get_communicator();
    dealii::ConditionalOStream      pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    // Match the initialized vector with the given vectors.
    bool found_match = false;
    for(unsigned int i = 0; i < data.e1_orientations->size(); i++)
    {
      if(e1_orientation->size() == (*data.e1_orientations)[i].size() and
         degree == data.degree_per_level[i])
      {
        pcout << "Filling vector of size " << e1_orientation->size() << " (degree = " << degree
              << ").\n";
        found_match = true;
        e1_orientation->copy_locally_owned_data_from((*data.e1_orientations)[i]);
        e2_orientation->copy_locally_owned_data_from((*data.e2_orientations)[i]);
      }
    }

    if(not found_match)
    {
      (*e1_orientation) = 0.0;
      e1_orientation->add(1.0);
      (*e1_orientation) = 0.0;
      e1_orientation->add(1.0);

      pcout << "Overwritten orientation vector of size " << e1_orientation->size()
            << " with dummy data.\n\n\n";
    }
    else
    {
      pcout << "|E1|_2 = " << e1_orientation->l2_norm() << "\n"
            << "|E2|_2 = " << e2_orientation->l2_norm() << "\n\n";
    }
#else
    AssertThrow(not orientation_vectors_provided,
                dealii::ExcMessage(
                  "You must link against ExaDG-Bio to enable user-defined material orientations."));
#endif

    e1_orientation->update_ghost_values();
    e2_orientation->update_ghost_values();
  }

  // Set the coefficients on the integration point level.
  VectorType dummy;
  matrix_free.cell_loop(&IncompressibleFibrousTissue<dim, Number>::cell_loop_set_coefficients,
                        this,
                        dummy,
                        dummy);

  // Release vectors after initialization, since cell data is stored.
  if(orientation_vectors_provided)
  {
    e1_orientation.reset();
    e2_orientation.reset();
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
  IntegratorCell integrator_e1(matrix_free, dof_index, quad_index);
  IntegratorCell integrator_e2(matrix_free, dof_index, quad_index);

  // The material coordinate system is initialized constant in the entire domain.
  // Phi is the angle from the circumferential vector E_1 towards the longitudinal
  // vector E_2.
  vector                 M_3;
  std::vector<vector>    M_1(n_fiber_families), M_2(n_fiber_families);
  dealii::Tensor<1, dim> E_1_default, E_2_default;
  for(unsigned int d = 0; d < dim; d++)
  {
    Number const reciprocal_norm = 1.0 / std::sqrt(static_cast<Number>(dim));
    E_1_default[d]               = reciprocal_norm;
    E_2_default[d]               = reciprocal_norm;
  }

  {
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      M_1[i] = fiber_cos_phi[i] * E_1_default - fiber_sin_phi[i] * E_2_default;
      M_2[i] = fiber_sin_phi[i] * E_1_default + fiber_cos_phi[i] * E_2_default;
    }

    dealii::Tensor<1, dim> E_3 = cross_product_3d(E_1_default, E_2_default);
    M_3                        = E_3;
  }

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

    if(orientation_vectors_provided)
    {
      integrator_e1.reinit(cell);
      integrator_e1.read_dof_values(*e1_orientation);
      integrator_e1.evaluate(dealii::EvaluationFlags::values);
      integrator_e2.reinit(cell);
      integrator_e2.read_dof_values(*e2_orientation);
      integrator_e2.evaluate(dealii::EvaluationFlags::values);
    }

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

      // Update the fiber mean directions.
      if(orientation_vectors_provided)
      {
        vector E_1 = integrator_e1.get_value(q);
        vector E_2 = integrator_e2.get_value(q);

        // Guard a potential zero norm, since we want to allow for
        // non-normed orientation vectors given by the user.
        Number constexpr tol = 1e-10;
        scalar E_1_norm      = E_1.norm();
        scalar E_2_norm      = E_2.norm();

        // factor_default = norm < tol ? 1.0 : 0.0
        scalar E_1_default_factor =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(E_1_norm,
                                                                            scalar_one * tol,
                                                                            scalar_one,
                                                                            scalar_zero);

        scalar E_2_default_factor =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(E_2_norm,
                                                                            scalar_one * tol,
                                                                            scalar_one,
                                                                            scalar_zero);

        // Add tol to norm to make sure we do not get undefined behavior.
        E_1 = E_1_default_factor * E_1_default +
              ((scalar_one - E_1_default_factor) / (E_1_norm + tol)) * E_1;
        E_2 = E_2_default_factor * E_2_default +
              ((scalar_one - E_2_default_factor) / (E_2_norm + tol)) * E_2;

        for(unsigned int i = 0; i < n_fiber_families; i++)
        {
          M_1[i] = fiber_cos_phi[i] * E_1 - fiber_sin_phi[i] * E_2;
          M_2[i] = fiber_sin_phi[i] * E_1 + fiber_cos_phi[i] * E_2;
        }
      }
      else
      {
        // The mean fiber directions are never updated after initialization
        // with the default E1 and E2 vectors.
      }

      // Fill the fiber orientation vectors or the structure tensor directly
      // based on the Euclidean material coordinate system and correct all
      // corresponding stored data initialized wrongly before.
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
        // Fiber families have identical M_3 since they lie in a plane normal to it.
        M_3 = cross_product_3d(M_1[0], M_2[0]);

        for(unsigned int i = 0; i < n_fiber_families; i++)
        {
          fiber_direction_M_1[i].set_coefficient_cell(cell, q, M_1[i]);

          // clang-format off
          tensor const H_i = fiber_H_11 * outer_product(M_1[i], M_1[i])
          	  	  	  	   + fiber_H_22 * outer_product(M_2[i], M_2[i])
          	  	           + fiber_H_33 * outer_product(M_3, M_3);
    	  fiber_structure_tensor[i].set_coefficient_cell(cell, q, H_i);
          // clang-format on

          // Update the products with the actual structure tensor.
          if(spatial_integration)
          {
            H_i_times_C_coefficients[i].set_coefficient_cell(cell, q, H_i /* * I */);
            C_times_H_i_coefficients[i].set_coefficient_cell(cell, q, /* I * */ H_i);
          }
        }
      }
    }
  }
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number>::get_c1(scalar const &     Jm1,
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
IncompressibleFibrousTissue<dim, Number>::get_c2(scalar const &     Jm1,
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
IncompressibleFibrousTissue<dim, Number>::get_J_pow(scalar const &     Jm1,
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
IncompressibleFibrousTissue<dim, Number>::get_structure_tensor(vector const &     M_1,
                                                               vector const &     M_2,
                                                               unsigned int const i,
                                                               unsigned int const cell,
                                                               unsigned int const q) const
{
  tensor H;

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

  return H;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
IncompressibleFibrousTissue<dim, Number>::get_fiber_switch(vector const & M_1,
                                                           tensor const & C) const
{
  // fiber_switch = I_i_star < 1 ? 0.0 : 1.0
  scalar const I_i = scalar_product(outer_product(M_1, M_1), C);

  scalar fiber_switch = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
    I_i, scalar_one, scalar_zero, scalar_one);

  return fiber_switch;
}

template<int dim, typename Number>
void
IncompressibleFibrousTissue<dim, Number>::do_set_cell_linearization_data(
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

    tensor const C = transpose(F) * F;

    tensor const E = get_E_scaled<dim, Number, Number>(Grad_d_lin, 1.0, stable_formulation);
    scalar const c1 =
      get_c1(Jm1, J_pow, E, shear_modulus_stored, true /* force_evaluation */, cell, q);
    scalar const c2 =
      get_c2(Jm1, J_pow, E, shear_modulus_stored, true /* force_evaluation */, cell, q);

    c1_coefficients.set_coefficient_cell(cell, q, c1);
    c2_coefficients.set_coefficient_cell(cell, q, c2);

    // Set scalar linearization data for fiber contribution.
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      vector const M_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
      vector       M_2_cache_level_0_1;
      if(cache_level < 2)
      {
        M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      }

      // Compute or load structure tensor.
      tensor const H_i = get_structure_tensor(M_1, M_2_cache_level_0_1, i, cell, q);

      scalar const E_i = scalar_product(H_i, C - I);
      E_i_coefficients[i].set_coefficient_cell(cell, q, E_i);

      scalar const fiber_switch = get_fiber_switch(M_1, C);
      scalar const c3           = (fiber_switch * 2.0 * fiber_k_1) * exp(fiber_k_2 * E_i * E_i);
      c3_coefficients[i].set_coefficient_cell(cell, q, c3);
    }

    if(cache_level > 1)
    {
      if(spatial_integration)
      {
        tensor const tau_lin =
          this->kirchhoff_stress(Grad_d_lin, cell, q, true /* force_evaluation */);
        kirchhoff_stress_coefficients.set_coefficient_cell(cell, q, tau_lin);

        C_coefficients.set_coefficient_cell(cell, q, C);

        // Set tensorial linearization data for fiber contribution.
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
      S = get_E_scaled<dim, Number, Number>(gradient_displacement,
                                            1.0,
                                            false /* stable_formulation */);
      scalar const c1 =
        get_c1(Jm1, J_pow, S /* E */, shear_modulus_stored, force_evaluation, cell, q);

      S = invert(C) * c1;
      add_scaled_identity(S, shear_modulus_stored * J_pow);
    }

    // Add fiber contribution.
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      // Compute or load fiber switch.
      vector M_1_cache_level_0_1, M_2_cache_level_0_1;
      if(cache_level < 2)
      {
        M_1_cache_level_0_1 = fiber_direction_M_1[i].get_coefficient_cell(cell, q);
        M_2_cache_level_0_1 = fiber_direction_M_2[i].get_coefficient_cell(cell, q);
      }
      else
      {
        // Dummy M_1 and M_2 sufficient.
      }

      // Compute or load structure tensor.
      tensor const H_i = get_structure_tensor(M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

      // Compute or load fiber strain-like quantity.
      scalar E_i;
      if(cache_level == 0)
      {
        E_i = scalar_product(H_i, C - I);
      }
      else
      {
        E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      }

      // Compute or load c3 coefficient.
      scalar c3;
      if(cache_level == 0)
      {
        scalar const fiber_switch = get_fiber_switch(M_1_cache_level_0_1, C);
        c3                        = (fiber_switch * 2.0 * fiber_k_1) * exp(fiber_k_2 * E_i * E_i);
      }
      else
      {
        c3 = c3_coefficients[i].get_coefficient_cell(cell, q);
      }

      S += (c3 * E_i) * H_i;
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
    // Compute or load fiber switch.
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
      // Dummy M_1, M_2 and C sufficient.
    }

    // Compute or load structure tensor.
    tensor const H_i = get_structure_tensor(M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

    // Compute or load fiber strain-like quantity.
    scalar E_i;
    if(cache_level == 0)
    {
      E_i = scalar_product(H_i, C_cache_level_0_1 - I);
    }
    else
    {
      E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
    }

    // Compute or load c3 coefficient.
    scalar c3;
    if(cache_level == 0)
    {
      scalar const fiber_switch = get_fiber_switch(M_1_cache_level_0_1, C_cache_level_0_1);

      c3 = (fiber_switch * 2.0 * fiber_k_1) * exp(fiber_k_2 * E_i * E_i);
    }
    else
    {
      c3 = c3_coefficients[i].get_coefficient_cell(cell, q);
    }

    Dd_S += H_i * (c3 * (2.0 * fiber_k_2 * E_i * E_i + 1.0) *
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

    // Add fiber contribution.
    for(unsigned int i = 0; i < n_fiber_families; i++)
    {
      // Compute or load fiber switch.
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
        // Dummy M_1, M_2 and C sufficient.
      }

      // Compute or load structure tensor.
      tensor const H_i = get_structure_tensor(M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

      // Compute or load fiber strain-like quantity.
      scalar E_i;
      if(cache_level == 0)
      {
        E_i = scalar_product(H_i, C_cache_level_0_1 - I);
      }
      else
      {
        E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
      }

      // Compute or load c3 coefficient.
      scalar c3;
      if(cache_level == 0)
      {
        scalar const fiber_switch = get_fiber_switch(M_1_cache_level_0_1, C_cache_level_0_1);
        c3                        = (fiber_switch * 2.0 * fiber_k_1) * exp(fiber_k_2 * E_i * E_i);
      }
      else
      {
        c3 = c3_coefficients[i].get_coefficient_cell(cell, q);
      }

      // Add terms in non-push-forwarded form.
      tau += (c3 * E_i) * H_i;
    }

    scalar const J_pow = get_J_pow(Jm1, force_evaluation, cell, q);

    tensor E;
    if(cache_level == 0 or force_evaluation or stable_formulation)
    {
      E = get_E_scaled<dim, Number, Number>(gradient_displacement, 1.0, stable_formulation);
    }
    else
    {
      // Dummy E sufficient.
    }

    // tau holds fiber terms in non-push-forwarded form, i.e.,
    // sum_(j=4,6) c_3 * E_i * H_i
    // to apply the push-forward only once for the sum of all terms.
    if(stable_formulation)
    {
      // Push forward the fiber contribution.
      tau = F * tau * transpose(F);

      // Add remaining terms.
      tau += (2.0 * shear_modulus_stored * J_pow) * E;

      add_scaled_identity<dim, Number>(tau,
                                       (-2.0 * one_third * shear_modulus_stored * J_pow) *
                                           trace(E) +
                                         0.5 * bulk_modulus *
                                           get_JJm1<Number>(Jm1, stable_formulation));
    }
    else
    {
      scalar const c1 = get_c1(Jm1, J_pow, E, shear_modulus_stored, force_evaluation, cell, q);

      // Add isochoric term, i.e.,
      // mu * J^(-2/3) * I
      // to apply push forward in next step
      add_scaled_identity(tau, shear_modulus_stored * J_pow);

      // Push forward the summed terms.
      tau = F * tau * transpose(F);

      // Add the remaining term.
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
IncompressibleFibrousTissue<dim, Number>::contract_with_J_times_C(
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

  // Add fiber contribution.
  for(unsigned int i = 0; i < n_fiber_families; i++)
  {
    // Compute or load fiber switch.
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

    // Compute or load structure tensor.
    tensor const H_i = get_structure_tensor(M_1_cache_level_0_1, M_2_cache_level_0_1, i, cell, q);

    // Compute or load fiber strain-like quantity.
    scalar E_i;
    if(cache_level == 0)
    {
      E_i = scalar_product(H_i, C - I);
    }
    else
    {
      E_i = E_i_coefficients[i].get_coefficient_cell(cell, q);
    }

    // Compute or load c3 coefficient.
    scalar c3;
    if(cache_level == 0)
    {
      scalar const fiber_switch = get_fiber_switch(M_1_cache_level_0_1, C);
      c3                        = (fiber_switch * 2.0 * fiber_k_1) * exp(fiber_k_2 * E_i * E_i);
    }
    else
    {
      c3 = c3_coefficients[i].get_coefficient_cell(cell, q);
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

    result += (2.0 * c3 * (2.0 * fiber_k_2 * E_i * E_i + 1.0) *
               scalar_product(H_i_times_C, symmetric_gradient_increment)) *
              C_times_H_i;
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
