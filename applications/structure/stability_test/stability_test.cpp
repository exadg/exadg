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

// C++
#include <stdlib.h>
#include <algorithm>
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/structure/material/library/compressible_neo_hookean.h>
#include <exadg/structure/material/library/incompressible_fibrous_tissue.h>
#include <exadg/structure/material/library/incompressible_neo_hookean.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/material/material.h>
#include <exadg/utilities/enum_utilities.h>

namespace ExaDG
{
namespace StabilityTest
{
using namespace ExaDG::Structure;

template<int dim, typename Number>
std::shared_ptr<Material<dim, Number>>
setup_material(MaterialType                            material_type,
               bool const                              spatial_integration,
               bool const                              stable_formulation,
               dealii::MatrixFree<dim, Number> const & matrix_free)
{
  // Construct Material objects.
  std::shared_ptr<Material<dim, Number>> material;

  // Dummy objects for initialization.
  unsigned int constexpr dof_index  = 0;
  unsigned int constexpr quad_index = 0;
  bool constexpr large_deformation  = true;
  unsigned int constexpr check_type = 0;

  if(material_type == MaterialType::StVenantKirchhoff)
  {
    // St.-Venant-Kirchhoff.
    Type2D const two_dim_type = Type2D::PlaneStress;
    double const nu           = 0.3;
    double const E_modul      = 200.0;

    StVenantKirchhoffData<dim> const data(MaterialType::StVenantKirchhoff,
                                          E_modul,
                                          nu,
                                          two_dim_type);

    material = std::make_shared<StVenantKirchhoff<dim, Number>>(
      matrix_free, dof_index, quad_index, data, large_deformation, check_type);
  }
  else
  {
    AssertThrow(dim == 3,
                dealii::ExcMessage("These material models are implemented for dim == 3 only."));

    Type2D constexpr two_dim_type          = Type2D::Undefined;
    bool constexpr force_material_residual = false;
    unsigned int constexpr cache_level     = 0;

    if(material_type == MaterialType::CompressibleNeoHookean)
    {
      // Compressible neo-Hookean.
      double const shear_modulus = 1.0e2;
      double const nu            = 0.3;
      double const lambda        = shear_modulus * 2.0 * nu / (1.0 - 2.0 * nu);

      CompressibleNeoHookeanData<dim> const data(MaterialType::CompressibleNeoHookean,
                                                 shear_modulus,
                                                 lambda,
                                                 two_dim_type);

      material = std::make_shared<CompressibleNeoHookean<dim, Number>>(matrix_free,
                                                                       dof_index,
                                                                       quad_index,
                                                                       data,
                                                                       spatial_integration,
                                                                       force_material_residual,
                                                                       check_type,
                                                                       stable_formulation,
                                                                       cache_level);
    }
    else if(material_type == MaterialType::IncompressibleNeoHookean)
    {
      // Incompressible neo-Hookean.
      double constexpr shear_modulus = 62.1e3;
      double constexpr nu            = 0.49;
      double constexpr bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      IncompressibleNeoHookeanData<dim> const data(MaterialType::IncompressibleNeoHookean,
                                                   shear_modulus,
                                                   bulk_modulus,
                                                   two_dim_type);

      material = std::make_shared<IncompressibleNeoHookean<dim, Number>>(matrix_free,
                                                                         dof_index,
                                                                         quad_index,
                                                                         data,
                                                                         spatial_integration,
                                                                         force_material_residual,
                                                                         check_type,
                                                                         stable_formulation,
                                                                         cache_level);
    }
    else if(material_type == MaterialType::IncompressibleFibrousTissue)
    {
      // Incompressible fibrous tissue.
      double constexpr shear_modulus = 62.1e3;
      double constexpr nu            = 0.49;
      double constexpr bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      // Parameters corresponding to aortic tissue might be found in
      // [Weisbecker et al., J Mech Behav Biomed Mater 12, 2012] or
      // [Rolf-Pissarczyk et al., Comput Methods Appl Mech Eng 373, 2021].
      // a = 3.62, b = 34.3 for medial tissue lead to the H_ii below,
      // while the k_1 coefficient is scaled relative to the shear modulus
      // (for medial tissue, e.g., 62.1 kPa).
      double const fiber_angle_phi_in_degree = 27.47;                            // [deg]
      double const fiber_H_11                = 0.9168;                           // [-]
      double const fiber_H_22                = 0.0759;                           // [-]
      double const fiber_H_33                = 0.0073;                           // [-]
      double const fiber_k_1                 = 1.4e3 * (shear_modulus / 62.1e3); // [Pa]
      double const fiber_k_2                 = 22.1;                             // [-]

      // Read the orientation files from binary format.
      typedef typename IncompressibleFibrousTissueData<dim>::VectorType VectorType;

      std::vector<VectorType>   e1_orientation;
      std::vector<VectorType>   e2_orientation;
      std::vector<unsigned int> degree_per_level;

      std::shared_ptr<std::vector<VectorType> const> e1_ori, e2_ori;
      e1_ori = std::make_shared<std::vector<VectorType> const>(e1_orientation);
      e2_ori = std::make_shared<std::vector<VectorType> const>(e2_orientation);

      IncompressibleFibrousTissueData<dim> const data(MaterialType::IncompressibleFibrousTissue,
                                                      shear_modulus,
                                                      bulk_modulus,
                                                      fiber_angle_phi_in_degree,
                                                      fiber_H_11,
                                                      fiber_H_22,
                                                      fiber_H_33,
                                                      fiber_k_1,
                                                      fiber_k_2,
                                                      nullptr /* e1_ori */,
                                                      nullptr /* e2_ori */,
                                                      {} /* degree_per_level */,
                                                      0.0 /* point_match_tol */,
                                                      two_dim_type);

      material = std::make_shared<IncompressibleFibrousTissue<dim, Number>>(matrix_free,
                                                                            dof_index,
                                                                            quad_index,
                                                                            data,
                                                                            spatial_integration,
                                                                            force_material_residual,
                                                                            check_type,
                                                                            stable_formulation,
                                                                            cache_level);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Material model not implemented."));
    }
  }

  return material;
}

template<typename Number>
std::vector<Number>
logspace(Number const & min_scale, Number const & max_scale, unsigned int const n_entries)
{
  Number const exponent_start = std::log10(min_scale);
  Number const exponent_end   = std::log10(max_scale);

  AssertThrow(n_entries > 2,
              dealii::ExcMessage("Constructing less than 3 data points does not make sense."));
  Number const exponent_delta =
    (exponent_end - exponent_start) / (static_cast<Number>(n_entries) - 1.0);

  std::vector<Number> vector(n_entries);
  Number              exponent = exponent_start;
  for(unsigned int i = 0; i < n_entries; ++i)
  {
    vector[i] = std::pow(10.0, exponent);
    exponent += exponent_delta;
  }
  return vector;
}

template<int dim, typename Number>
dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
get_random_tensor(Number scale)
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> random_tensor;

  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = 0; j < dim; ++j)
    {
      // Pseudo-random number, but in the interval [0.1, 0.9] to
      // keep the order of magnitude unchanged.
      Number const random_number_0c0_1c0 =
        static_cast<Number>(std::rand()) / static_cast<Number>(RAND_MAX);
      Number const random_number_0c1_0c9 = 0.1 + (random_number_0c0_1c0 / 0.8);
      random_tensor[i][j]                = scale * random_number_0c1_0c9;
    }
  }

  return random_tensor;
}

template<int dim, typename NumberIn, typename NumberOut>
dealii::Tensor<2, dim, dealii::VectorizedArray<NumberOut>>
tensor_cast_shallow(dealii::Tensor<2, dim, dealii::VectorizedArray<NumberIn>> const & tensor_in)
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<NumberOut>> tensor_out;

  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = 0; j < dim; ++j)
    {
      // Copy and cast only the very first entry from `tensor_in` to `tensor_out`.
      tensor_out[i][j][0] = static_cast<NumberOut>(tensor_in[i][j][0]);
    }
  }

  return tensor_out;
}

template<int dim, typename Number>
std::vector<dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>>
evaluate_material(
  Material<dim, Number> const &                                   material,
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_increment,
  bool const                                                      spatial_integration)
{
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> Tensor;

  // Dummy parameters.
  unsigned int constexpr cell = 0;
  unsigned int constexpr q    = 0;

  // No that this is redundant, since cache_level has to be
  // zero anyways to not access integration point storage.
  bool constexpr force_evaluation = true;
  std::vector<Tensor> evaluation(2);
  if(spatial_integration)
  {
    evaluation[0] = material.kirchhoff_stress(gradient_displacement, cell, q, force_evaluation);
    Tensor const symmetric_gradient_increment =
      0.5 * (gradient_increment + transpose(gradient_increment));
    evaluation[1] = material.contract_with_J_times_C(symmetric_gradient_increment,
                                                     gradient_displacement,
                                                     cell,
                                                     q);
  }
  else
  {
    evaluation[0] =
      material.second_piola_kirchhoff_stress(gradient_displacement, cell, q, force_evaluation);
    evaluation[1] = material.second_piola_kirchhoff_stress_displacement_derivative(
      gradient_increment, gradient_displacement, cell, q);
  }

  return evaluation;
}

} // namespace StabilityTest
} // namespace ExaDG

int
main(int argc, char ** argv)
{
  using namespace ExaDG::StabilityTest;

  unsigned int constexpr dim = 3;

  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<double>> tensor_double;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<float>>  tensor_float;

  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    // Gradient and increment scale.
    unsigned int constexpr n_points_over_log_scale = 1e3;
    std::vector<double> grad_u_scale               = logspace(1e-8, 1e+2, n_points_over_log_scale);
    double constexpr h_e                           = 1e-3;
    double constexpr grad_increment_scale          = 1.0 / (h_e * h_e);

    // Setup dummy MatrixFree object in case the material
    // model stores some data even for cache_level 0.
    dealii::MatrixFree<dim, float>  matrix_free_float;
    dealii::MatrixFree<dim, double> matrix_free_double;

    dealii::Triangulation<dim> triangulation;
    dealii::GridGenerator::hyper_cube(triangulation, 0., 1.);
    dealii::FE_Q<dim> const fe_q(1);
    dealii::MappingQ1<dim>  mapping;
    dealii::DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe_q);
    {
      // For MatrixFree<dim, double> object.
      dealii::AffineConstraints<double> empty_constraints_double;
      empty_constraints_double.close();

      typename dealii::MatrixFree<dim, double>::AdditionalData additional_data_double;
      // additional_data_double.mapping_update_flags =
      //   (update_gradients | update_JxW_values | update_quadrature_points);
      matrix_free_double.reinit(mapping,
                                dof_handler,
                                empty_constraints_double,
                                dealii::QGauss<1>(2),
                                additional_data_double);

      // For MatrixFree<dim, float> object.
      dealii::AffineConstraints<float> empty_constraints_float;
      empty_constraints_float.close();

      typename dealii::MatrixFree<dim, float>::AdditionalData additional_data_float;
      // additional_data_float.mapping_update_flags =
      //   (update_gradients | update_JxW_values | update_quadrature_points);
      matrix_free_float.reinit(
        mapping, dof_handler, empty_constraints_float, dealii::QGauss<1>(2), additional_data_float);
    }

    std::vector<MaterialType> material_type_vec{MaterialType::StVenantKirchhoff,
                                                MaterialType::CompressibleNeoHookean,
                                                MaterialType::IncompressibleNeoHookean,
                                                MaterialType::IncompressibleFibrousTissue};
    std::vector<bool>         spatial_integration_vec{false, true};
    std::vector<bool>         stable_formulation_vec{false, true};

    for(MaterialType const material_type : material_type_vec)
    {
      for(bool const spatial_integration : spatial_integration_vec)
      {
        for(bool const stable_formulation : stable_formulation_vec)
        {
          // Skip some cases not implemented for the STVK model.
          if(material_type == MaterialType::StVenantKirchhoff &&
             (spatial_integration || not stable_formulation))
          {
            continue;
          }

          pcout << "  Material model : " << ExaDG::Utilities::enum_to_string(material_type) << "\n"
                << "  spatial_integration = " << spatial_integration << "\n"
                << "  stable_formulation = " << stable_formulation << "\n";

          // Setup material objects in float and double precision for comparison.
          std::shared_ptr<Material<dim, double>> material_double = setup_material<dim, double>(
            material_type, spatial_integration, stable_formulation, matrix_free_double);
          std::shared_ptr<Material<dim, float>> material_float = setup_material<dim, float>(
            material_type, spatial_integration, stable_formulation, matrix_free_float);

          // Evaluate the residual and derivative for all given scalings.
          std::vector<std::vector<double>> relative_error_samples(2);
          relative_error_samples[0].resize(n_points_over_log_scale);
          relative_error_samples[1].resize(n_points_over_log_scale);

          for(unsigned int i = 0; i < n_points_over_log_scale; ++i)
          {
            // Repeat the experiment with `n_samples`, store the highest error sampled.
            unsigned int n_samples = 2e2;
            for(unsigned int j = 0; j < n_samples; ++j)
            {
              // Generate pseudo-random gradient of the displacement field.
              tensor_float const gradient_displacement_float =
                get_random_tensor<dim, float>(grad_u_scale[i]);
              tensor_float const gradient_increment_float =
                get_random_tensor<dim, float>(grad_increment_scale);

              // Deep copy the tensor entries casting to *representable*
              // double type to have the *same* random number.
              tensor_double const gradient_displacement_double =
                tensor_cast_shallow<dim, float /*NumberIn*/, double /*NumberOut*/>(
                  gradient_displacement_float);
              tensor_double const gradient_increment_double =
                tensor_cast_shallow<dim, float /*NumberIn*/, double /*NumberOut*/>(
                  gradient_increment_float);

              // Compute stresses and derivatives.
              std::vector<tensor_double> const evaluation_double =
                evaluate_material<dim, double>(*material_double,
                                               gradient_displacement_double,
                                               gradient_increment_double,
                                               spatial_integration);

              std::vector<tensor_float> const evaluation_float =
                evaluate_material<dim, float>(*material_float,
                                              gradient_displacement_float,
                                              gradient_increment_float,
                                              spatial_integration);

              // Store the current sample, if it gives the largest relative error.
              AssertThrow(evaluation_double.size() == 2,
                          dealii::ExcMessage("Expecting stress and derivative."));
              for(unsigned int k = 0; k < evaluation_double.size(); ++k)
              {
                // Cast first entry of computed tensor to double from *representable* float.
                tensor_double diff_evaluation =
                  tensor_cast_shallow<dim, float /*NumberIn*/, double /*NumberOut*/>(
                    evaluation_float[k]);
                diff_evaluation -= evaluation_double[k];

                // Maximum relative error in the |tensor|_inf norm.
                bool constexpr take_max_err = true;
                double rel_norm_evaluation  = take_max_err ? 1e-20 : 1e+20;
                for(unsigned int l = 0; l < dim; ++l)
                {
                  for(unsigned int m = 0; m < dim; ++m)
                  {
                    double const rel_norm = std::abs((diff_evaluation[l][m][0] + 1e-40) /
                                                     (evaluation_double[k][l][m][0] + 1e-20));
                    if(take_max_err)
                    {
                      rel_norm_evaluation = std::max(rel_norm_evaluation, rel_norm);
                    }
                    else
                    {
                      rel_norm_evaluation = std::min(rel_norm_evaluation, rel_norm);
                    }
                  }
                }

                // Extract only the first entry, since all entries in the
                // dealii::VectorizedArray<Number> have the same value.
                relative_error_samples[k][i] =
                  std::max(relative_error_samples[k][i], rel_norm_evaluation);
              }
            }
          }

          // Output the results to file.
          std::string const file_name =
            "./stability_forward_test_spatial_integration_" + std::to_string(spatial_integration) +
            "_stable_formulation_" + std::to_string(stable_formulation) + +"_" +
            ExaDG::Utilities::enum_to_string(material_type) + ".txt";

          std::ofstream fstream;
          size_t const  fstream_buffer_size = 256 * 1024; // TODO measure if this has any effect.
          char          fstream_buffer[fstream_buffer_size];
          fstream.rdbuf()->pubsetbuf(fstream_buffer, fstream_buffer_size);
          fstream.open(file_name.c_str(), std::ios::trunc);

          fstream << "  relative errors in stress and stress derivative,\n"
                  << "  grad_increment_scale = 1 / h_e^2 , h_e = " << h_e << "\n"
                  << "  in |.| = max_(samples){ | T_f64 - T_f32 |inf / |T_f64|inf }\n"
                  << "  grad_u_scl    |stress|      |Jacobian|\n";

          for(unsigned int i = 0; i < n_points_over_log_scale; ++i)
          {
            fstream << "  " << std::scientific << std::setw(10) << grad_u_scale[i];
            for(unsigned int k = 0; k < relative_error_samples.size(); ++k)
            {
              fstream << "  " << std::scientific << std::setw(10) << relative_error_samples[k][i];
            }
            fstream << "\n";
          }
          fstream.close();
        }
      }
    }
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
