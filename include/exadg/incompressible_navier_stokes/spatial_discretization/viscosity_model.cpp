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

#include <exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::ViscosityModel()
  : matrix_free(nullptr)
{
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  dealii::Mapping<dim> const &                           mapping_in,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
  TurbulenceModelData const &                            turbulence_model_data_in,
  GeneralizedNewtonianModelData const &                  generalized_newtonian_model_data_in)
{
  matrix_free                      = &matrix_free_in;
  viscous_kernel                   = viscous_kernel_in;
  turbulence_model_data            = turbulence_model_data_in;
  generalized_newtonian_model_data = generalized_newtonian_model_data_in;

  if(use_turbulence_model == true)
  {
    calculate_filter_width(mapping_in);
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  calculate_viscosity(VectorType const & velocity) const
{
  VectorType dummy;

  matrix_free->loop(&This::cell_loop_set_coefficients,
                    &This::face_loop_set_coefficients,
                    &This::boundary_face_loop_set_coefficients,
                    this,
                    dummy,
                    velocity);
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const
{
  CellIntegratorU integrator(matrix_free,
                             turbulence_model_data.dof_index,
                             turbulence_model_data.quad_index);

  // containers needed dependent on template parameters
  scalar filter_width;
  scalar viscosity;

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    // get filter width for this cell
    if(use_turbulence_model == true)
    {
      filter_width = integrator.read_cell_data(this->filter_width_vector);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate needed quantities for this cell
      tensor velocity_gradient           = integrator.get_gradient(q);
      tensor symmetric_velocity_gradient = dealii::make_vectorized_array<Number>(0.5) *
                                           (velocity_gradient + transpose(velocity_gradient));
      scalar shear_rate_squared =
        dealii::make_vectorized_array<Number>(2.0) *
        scalar_product(symmetric_velocity_gradient, symmetric_velocity_gradient);

      if(use_generalized_newtonian_model == false)
      {
        viscosity =
          dealii::make_vectorized_array<Number>(turbulence_model_data.kinematic_viscosity);
      }
      else
      {
        set_generalized_newtonian_viscosity(shear_rate_squared, viscosity);
      }

      if(use_turbulence_model == true)
      {
        add_turbulent_viscosity(viscosity /*might use generalized Newtonian viscosity*/,
                                filter_width,
                                velocity_gradient,
                                symmetric_velocity_gradient,
                                shear_rate_squared,
                                turbulence_model_data.constant);
      }
      // set the coefficients
      viscous_kernel->set_coefficient_cell(cell, q, viscosity);
    }
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const & src,
                             Range const &      face_range) const
{
  FaceIntegratorU integrator_m(matrix_free,
                               true,
                               turbulence_model_data.dof_index,
                               turbulence_model_data.quad_index);
  FaceIntegratorU integrator_p(matrix_free,
                               false,
                               turbulence_model_data.dof_index,
                               turbulence_model_data.quad_index);

  // containers needed dependent on template parameters
  scalar filter_width;
  scalar filter_width_neighbor;
  scalar viscosity;
  scalar viscosity_neighbor;

  // loop over all interior faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.read_dof_values(src);
    integrator_p.read_dof_values(src);

    // we only need the gradient
    integrator_m.evaluate(dealii::EvaluationFlags::gradients);
    integrator_p.evaluate(dealii::EvaluationFlags::gradients);

    // get filter width for this cell and the neighbor
    if(use_turbulence_model == true)
    {
      filter_width          = integrator_m.read_cell_data(this->filter_width_vector);
      filter_width_neighbor = integrator_p.read_cell_data(this->filter_width_vector);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      // calculate needed quantities for both elements adjacent to the current face
      tensor velocity_gradient          = integrator_m.get_gradient(q);
      tensor velocity_gradient_neighbor = integrator_p.get_gradient(q);

      tensor symmetric_velocity_gradient = dealii::make_vectorized_array<Number>(0.5) *
                                           (velocity_gradient + transpose(velocity_gradient));
      tensor symmetric_velocity_gradient_neighbor =
        dealii::make_vectorized_array<Number>(0.5) *
        (velocity_gradient_neighbor + transpose(velocity_gradient_neighbor));

      scalar shear_rate_squared =
        dealii::make_vectorized_array<Number>(2.0) *
        scalar_product(symmetric_velocity_gradient, symmetric_velocity_gradient);
      scalar shear_rate_squared_neighbor =
        dealii::make_vectorized_array<Number>(2.0) *
        scalar_product(symmetric_velocity_gradient_neighbor, symmetric_velocity_gradient_neighbor);

      if(use_generalized_newtonian_model == false)
      {
        viscosity =
          dealii::make_vectorized_array<Number>(turbulence_model_data.kinematic_viscosity);
        viscosity_neighbor =
          dealii::make_vectorized_array<Number>(turbulence_model_data.kinematic_viscosity);
      }
      else
      {
        set_generalized_newtonian_viscosity(shear_rate_squared, viscosity);
        set_generalized_newtonian_viscosity(shear_rate_squared_neighbor, viscosity_neighbor);
      }

      if(use_turbulence_model == true)
      {
        add_turbulent_viscosity(viscosity /*might use generalized Newtonian viscosity*/,
                                filter_width,
                                velocity_gradient,
                                symmetric_velocity_gradient,
                                shear_rate_squared,
                                turbulence_model_data.constant);
        add_turbulent_viscosity(viscosity_neighbor /*might use generalized Newtonian viscosity*/,
                                filter_width_neighbor,
                                velocity_gradient_neighbor,
                                symmetric_velocity_gradient_neighbor,
                                shear_rate_squared_neighbor,
                                turbulence_model_data.constant);
      }

      // set the coefficients
      viscous_kernel->set_coefficient_face(face, q, viscosity);
      viscous_kernel->set_coefficient_face_neighbor(face, q, viscosity_neighbor);
    }
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  boundary_face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                                      VectorType &,
                                      VectorType const & src,
                                      Range const &      face_range) const
{
  FaceIntegratorU integrator(matrix_free,
                             true,
                             turbulence_model_data.dof_index,
                             turbulence_model_data.quad_index);

  // containers needed dependent on template parameters
  scalar filter_width;
  scalar viscosity;

  // loop over all boundary faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    // get filter width for this cell
    if(use_turbulence_model == true)
    {
      filter_width = integrator.read_cell_data(this->filter_width_vector);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate needed quantities for this face
      tensor velocity_gradient           = integrator.get_gradient(q);
      tensor symmetric_velocity_gradient = dealii::make_vectorized_array<Number>(0.5) *
                                           (velocity_gradient + transpose(velocity_gradient));
      scalar shear_rate_squared =
        dealii::make_vectorized_array<Number>(2.0) *
        scalar_product(symmetric_velocity_gradient, symmetric_velocity_gradient);

      if(use_generalized_newtonian_model == false)
      {
        viscosity =
          dealii::make_vectorized_array<Number>(turbulence_model_data.kinematic_viscosity);
      }
      else
      {
        set_generalized_newtonian_viscosity(shear_rate_squared, viscosity);
      }

      if(use_turbulence_model == true)
      {
        add_turbulent_viscosity(viscosity,
                                filter_width,
                                velocity_gradient,
                                symmetric_velocity_gradient,
                                shear_rate_squared,
                                turbulence_model_data.constant);
      }

      // set the coefficients
      viscous_kernel->set_coefficient_face(face, q, viscosity);
    }
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  calculate_filter_width(dealii::Mapping<dim> const & mapping)
{
  unsigned int n_cells = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();

  filter_width_vector.resize(n_cells);

  unsigned int const dof_index = turbulence_model_data.dof_index;

  dealii::QGauss<dim> quadrature(turbulence_model_data.degree + 1);

  dealii::FEValues<dim> fe_values(mapping,
                                  matrix_free->get_dof_handler(dof_index).get_fe(),
                                  quadrature,
                                  dealii::update_JxW_values);

  double one_over_degree_plus_one = 1.0 / double(turbulence_model_data.degree + 1);

  // loop over all cells
  for(unsigned int i = 0; i < n_cells; ++i)
  {
    for(unsigned int v = 0; v < matrix_free->n_active_entries_per_cell_batch(i); ++v)
    {
      typename dealii::DoFHandler<dim>::cell_iterator cell =
        matrix_free->get_cell_iterator(i, v, dof_index);
      fe_values.reinit(cell);

      // calculate cell volume
      double volume = 0.0;
      for(unsigned int q = 0; q < quadrature.size(); ++q)
      {
        volume += fe_values.JxW(q);
      }

      // h = V^{1/dim}
      double h = std::exp(std::log(volume) / (double)dim);

      // take polynomial degree of shape functions into account:
      // h/(k_u + 1)
      h *= one_over_degree_plus_one;

      filter_width_vector[i][v] = h;
    }
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  add_turbulent_viscosity(scalar &       viscosity,
                          scalar const & filter_width,
                          tensor const & velocity_gradient,
                          tensor const & symmetric_velocity_gradient,
                          scalar const & shear_rate_squared,
                          double const & model_constant) const
{
  switch(turbulence_model_data.turbulence_model)
  {
    case TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(turbulence_model_data.turbulence_model != TurbulenceEddyViscosityModel::Undefined,
                  dealii::ExcMessage("Parameter must be defined."));
      break;
    case TurbulenceEddyViscosityModel::Smagorinsky:
      smagorinsky_turbulence_model(filter_width, shear_rate_squared, model_constant, viscosity);
      break;
    case TurbulenceEddyViscosityModel::Vreman:
      vreman_turbulence_model(
        filter_width, velocity_gradient, symmetric_velocity_gradient, model_constant, viscosity);
      break;
    case TurbulenceEddyViscosityModel::WALE:
      wale_turbulence_model(
        filter_width, velocity_gradient, shear_rate_squared, model_constant, viscosity);
      break;
    case TurbulenceEddyViscosityModel::Sigma:
      sigma_turbulence_model(filter_width, symmetric_velocity_gradient, model_constant, viscosity);
      break;
    default:
      AssertThrow(
        false,
        dealii::ExcMessage(
          "This TurbulenceEddyViscosityModel is not implemented in viscosity_model.cpp ."));
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  smagorinsky_turbulence_model(scalar const & filter_width,
                               scalar const & shear_rate_squared,
                               double const & C,
                               scalar &       viscosity) const
{
  scalar factor = C * filter_width;
  factor *= factor;
  factor *= std::sqrt(shear_rate_squared);
  viscosity += factor;
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  vreman_turbulence_model(scalar const & filter_width,
                          tensor const & velocity_gradient,
                          tensor const & symmetric_velocity_gradient,
                          double const & C,
                          scalar &       viscosity) const
{
  scalar       velocity_gradient_norm_square = scalar_product(velocity_gradient, velocity_gradient);
  Number const tolerance                     = 1.0e-12;

  tensor tensor = dealii::make_vectorized_array<Number>(2.0) * symmetric_velocity_gradient;

  AssertThrow(dim == 3,
              dealii::ExcMessage(
                "Number of dimensions has to be dim==3 to evaluate Vreman turbulence model."));

  scalar B_gamma = +tensor[0][0] * tensor[1][1] - tensor[0][1] * tensor[0][1] +
                   tensor[0][0] * tensor[2][2] - tensor[0][2] * tensor[0][2] +
                   tensor[1][1] * tensor[2][2] - tensor[1][2] * tensor[1][2];

  scalar factor = C * filter_width;

  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); i++)
  {
    // If the norm of the velocity gradient tensor is zero, the subgrid-scale
    // viscosity is defined as zero, so we do nothing in that case.
    // Make sure that B_gamma[i] is larger than zero since we calculate
    // the square root of B_gamma[i].
    if(velocity_gradient_norm_square[i] > tolerance && B_gamma[i] > tolerance)
    {
      viscosity[i] += factor[i] * factor[i] *
                      std::exp(0.5 * std::log(B_gamma[i] / velocity_gradient_norm_square[i]));
    }
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  wale_turbulence_model(scalar const & filter_width,
                        tensor const & velocity_gradient,
                        scalar const & shear_rate_squared,
                        double const & C,
                        scalar &       viscosity) const
{
  scalar S_norm_square = dealii::make_vectorized_array<Number>(0.5) * shear_rate_squared;

  tensor square_gradient       = velocity_gradient * velocity_gradient;
  scalar trace_square_gradient = trace(square_gradient);

  tensor isotropic_tensor;
  for(unsigned int i = 0; i < dim; ++i)
  {
    isotropic_tensor[i][i] = 1.0 / 3.0 * trace_square_gradient;
  }

  tensor S_d =
    dealii::make_vectorized_array<Number>(0.5) * (square_gradient + transpose(square_gradient)) -
    isotropic_tensor;

  scalar S_d_norm_square = scalar_product(S_d, S_d);

  scalar D = dealii::make_vectorized_array<Number>(0.0);

  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); i++)
  {
    Number const tolerance = 1.e-12;
    if(S_d_norm_square[i] > tolerance)
    {
      D[i] = std::pow(S_d_norm_square[i], 1.5) /
             (std::pow(S_norm_square[i], 2.5) + std::pow(S_d_norm_square[i], 1.25));
    }
  }

  scalar factor = C * filter_width;

  viscosity += factor * factor * D;
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  sigma_turbulence_model(scalar const & filter_width,
                         tensor const & symmetric_velocity_gradient,
                         double const & C,
                         scalar &       viscosity) const
{
  AssertThrow(dim == 3,
              dealii::ExcMessage(
                "Number of dimensions has to be dim==3 to evaluate Sigma turbulence model."));

  /*
   *  Compute singular values manually using a self-contained method
   *  (see appendix in Nicoud et al. (2011)). This approach is more efficient
   *  than calculating eigenvalues or singular values using LAPACK routines.
   */
  scalar D = dealii::make_vectorized_array<Number>(0.0);

  tensor G = dealii::make_vectorized_array<Number>(2.0) * symmetric_velocity_gradient;

  scalar invariant1 = trace(G);
  scalar invariant2 = 0.5 * (invariant1 * invariant1 - trace(G * G));
  scalar invariant3 = determinant(G);

  for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  {
    // if trace(G) = 0, all eigenvalues (and all singular values) have to be zero
    // and hence G is also zero. Set D[n]=0 in that case.
    if(invariant1[n] > 1.0e-12)
    {
      Number alpha1 = invariant1[n] * invariant1[n] / 9.0 - invariant2[n] / 3.0;
      Number alpha2 = invariant1[n] * invariant1[n] * invariant1[n] / 27.0 -
                      invariant1[n] * invariant2[n] / 6.0 + invariant3[n] / 2.0;

      AssertThrow(alpha1 >= std::numeric_limits<double>::denorm_min() /*smallest positive value*/,
                  dealii::ExcMessage("alpha1 has to be larger than zero."));

      Number factor = alpha2 / std::pow(alpha1, 1.5);

      AssertThrow(std::abs(factor) <=
                    1.0 + 1.0e-12, /* we found that a larger tolerance (1e-8,1e-6,1e-4) might be
                                      necessary in some cases */
                  dealii::ExcMessage("Cannot compute arccos(value) if abs(value)>1.0."));

      // Ensure that the argument of arccos() is in the interval [-1,1].
      if(factor > 1.0)
        factor = 1.0;
      else if(factor < -1.0)
        factor = -1.0;

      Number alpha3 = 1.0 / 3.0 * std::acos(factor);

      dealii::Vector<Number> sv = dealii::Vector<Number>(dim);

      sv[0] = invariant1[n] / 3.0 + 2 * std::sqrt(alpha1) * std::cos(alpha3);
      sv[1] =
        invariant1[n] / 3.0 - 2 * std::sqrt(alpha1) * std::cos(dealii::numbers::PI / 3.0 + alpha3);
      sv[2] =
        invariant1[n] / 3.0 - 2 * std::sqrt(alpha1) * std::cos(dealii::numbers::PI / 3.0 - alpha3);

      // Calculate the square root only if the value is larger than zero.
      // Otherwise set sv to zero (this is reasonable since negative values will
      // only occur due to numerical errors).
      for(unsigned int d = 0; d < dim; ++d)
      {
        if(sv[d] > 0.0)
          sv[d] = std::sqrt(sv[d]);
        else
          sv[d] = 0.0;
      }

      Number const tolerance = 1.e-12;
      if(sv[0] > tolerance)
      {
        D[n] = (sv[2] * (sv[0] - sv[1]) * (sv[1] - sv[2])) / (sv[0] * sv[0]);
      }
    }
  }

  /*
   * The singular values of the velocity gradient g = grad(u) are
   * the square root of the eigenvalues of G = g^T * g.
   */

  //    scalar D_copy = D; // save a copy in order to verify the correctness of
  //    the computation for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size();
  //    n++)
  //    {
  //      LAPACKFullMatrix<Number> G_local = LAPACKFullMatrix<Number>(dim);
  //
  //      for(unsigned int i = 0; i < dim; i++)
  //      {
  //        for(unsigned int j = 0; j < dim; j++)
  //        {
  //          G_local(i,j) = G[i][j][n];
  //        }
  //      }
  //
  //      G_local.compute_eigenvalues();
  //
  //      std::list<Number> ev_list;
  //
  //      for(unsigned int l = 0; l < dim; l++)
  //      {
  //        ev_list.push_back(std::abs(G_local.eigenvalue(l)));
  //      }
  //
  //      // This sorts the list in ascending order, beginning with the smallest eigenvalue.
  //      ev_list.sort();
  //
  //      dealii::Vector<Number> ev = dealii::Vector<Number>(dim);
  //      typename std::list<Number>::reverse_iterator it;
  //      unsigned int k;
  //
  //      // Write values in vector "ev" and reverse the order so that we
  //      // ev[0] corresponds to the largest eigenvalue.
  //      for(it = ev_list.rbegin(), k=0; it != ev_list.rend() && k<dim; ++it, ++k)
  //      {
  //        ev[k] = std::sqrt(*it);
  //      }
  //
  //      Number const tolerance = 1.e-12;
  //      if(ev[0] > tolerance)
  //      {
  //        D[n] = (ev[2]*(ev[0]-ev[1])*(ev[1]-ev[2]))/(ev[0]*ev[0]);
  //      }
  //    }
  //
  //    // make sure that both variants yield the same result
  //    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  //    {
  //      AssertThrow(std::abs(D[n]-D_copy[n])<1.e-5,dealii::ExcMessage("Calculation of singular
  //      values is incorrect."));
  //    }


  /*
   *  Alternatively, compute singular values directly using SVD.
   */
  //    scalar D_copy2 = D; // save a copy in order to verify the correctness of
  //    the computation D = dealii::make_vectorized_array<Number>(0.0);
  //
  //    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  //    {
  //      LAPACKFullMatrix<Number> gradient = LAPACKFullMatrix<Number>(dim);
  //      for(unsigned int i = 0; i < dim; i++)
  //      {
  //        for(unsigned int j = 0; j < dim; j++)
  //        {
  //          gradient(i,j) = velocity_gradient[i][j][n];
  //        }
  //      }
  //      gradient.compute_svd();
  //
  //      dealii::Vector<Number> sv = dealii::Vector<Number>(dim);
  //      for(unsigned int i=0;i<dim;++i)
  //      {
  //        sv[i] = gradient.singular_value(i);
  //      }
  //
  //      Number const tolerance = 1.e-12;
  //      if(sv[0] > tolerance)
  //      {
  //        D[n] = (sv[2]*(sv[0]-sv[1])*(sv[1]-sv[2]))/(sv[0]*sv[0]);
  //      }
  //    }
  //
  //    // make sure that both variants yield the same result
  //    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); n++)
  //    {
  //      AssertThrow(std::abs(D[n]-D_copy2[n])<1.e-5,dealii::ExcMessage("Calculation of singular
  //      values is incorrect."));
  //    }

  // add turbulent eddy-viscosity to laminar viscosity
  scalar factor = C * filter_width;
  viscosity += factor * factor * D;
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  set_generalized_newtonian_viscosity(scalar const & shear_rate_squared, scalar & viscosity) const
{
  switch(generalized_newtonian_model_data.generalized_newtonian_model)
  {
    case GeneralizedNewtonianModel::Undefined:
      AssertThrow(generalized_newtonian_model_data.generalized_newtonian_model !=
                    GeneralizedNewtonianModel::Undefined,
                  dealii::ExcMessage("parameter must be defined"));
      break;
    case GeneralizedNewtonianModel::GeneralizedCarreauYasuda:
      generalized_carreau_yasuda_generalized_newtonian_model(shear_rate_squared, viscosity);
      break;
    case GeneralizedNewtonianModel::Carreau:
      carreau_generalized_newtonian_model(shear_rate_squared, viscosity);
      break;
    case GeneralizedNewtonianModel::Cross:
      cross_generalized_newtonian_model(shear_rate_squared, viscosity);
      break;
    case GeneralizedNewtonianModel::SimplifiedCross:
      simplified_cross_generalized_newtonian_model(shear_rate_squared, viscosity);
      break;
    case GeneralizedNewtonianModel::PowerLaw:
      power_law_generalized_newtonian_model(shear_rate_squared, viscosity);
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage(
                    "This GeneralizedNewtonianModel is not implemented in viscosity_model.cpp ."));
  }
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  generalized_carreau_yasuda_generalized_newtonian_model(scalar const & shear_rate_squared,
                                                         scalar &       viscosity) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  viscosity = shear_rate_squared * dealii::make_vectorized_array<Number>(
                                     generalized_newtonian_model_data.lambda_squared);
  viscosity =
    std::pow(viscosity,
             dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.a * 0.5));
  viscosity += dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.kappa);
  viscosity =
    std::pow(viscosity,
             dealii::make_vectorized_array<Number>((generalized_newtonian_model_data.n - 1.0) /
                                                   generalized_newtonian_model_data.a));

  viscosity *= dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_upper_limit -
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
  viscosity += dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  carreau_generalized_newtonian_model(scalar const & shear_rate_squared, scalar & viscosity) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 1 and a = 2
  // eta = eta_oo + (eta_0 - eta_oo) * [1 + l^2 * y^2]^[(n-1)/2]
  viscosity = shear_rate_squared * dealii::make_vectorized_array<Number>(
                                     generalized_newtonian_model_data.lambda_squared);

  // here we can skip one std::pow() call
  viscosity += dealii::make_vectorized_array<Number>(1.0);
  viscosity = std::pow(viscosity,
                       dealii::make_vectorized_array<Number>(
                         (generalized_newtonian_model_data.n - 1.0) / 2.0));

  viscosity *= dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_upper_limit -
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
  viscosity += dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  cross_generalized_newtonian_model(scalar const & shear_rate_squared, scalar & viscosity) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 1 and n = 1 - a
  // eta = eta_oo + (eta_0 - eta_oo) * [1 + (l * y)^a]^(-1)
  viscosity = shear_rate_squared * dealii::make_vectorized_array<Number>(
                                     generalized_newtonian_model_data.lambda_squared);
  viscosity =
    std::pow(viscosity,
             dealii::make_vectorized_array<Number>((generalized_newtonian_model_data.a) * 0.5));
  viscosity += dealii::make_vectorized_array<Number>(1.0);

  // here we can replace the second std::pow() call with a ^(-1)
  viscosity = dealii::make_vectorized_array<Number>(1.0) / viscosity;

  viscosity *= dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_upper_limit -
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
  viscosity += dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  simplified_cross_generalized_newtonian_model(scalar const & shear_rate_squared,
                                               scalar &       viscosity) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 1, a = 1 and n = 0
  // eta = eta_oo + (eta_0 - eta_oo) * [1 + l * y]^(-1)
  viscosity = shear_rate_squared * dealii::make_vectorized_array<Number>(
                                     generalized_newtonian_model_data.lambda_squared);

  // here we can use std::sqrt() instead of std::pow()
  viscosity = std::sqrt(viscosity);
  viscosity += dealii::make_vectorized_array<Number>(1.0);

  // here we can replace the second std::pow() call with a ^(-1)
  viscosity = dealii::make_vectorized_array<Number>(1.0) / viscosity;

  viscosity *= dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_upper_limit -
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
  viscosity += dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_lower_limit);
}

template<int dim, typename Number, bool use_turbulence_model, bool use_generalized_newtonian_model>
void
ViscosityModel<dim, Number, use_turbulence_model, use_generalized_newtonian_model>::
  power_law_generalized_newtonian_model(scalar const & shear_rate_squared, scalar & viscosity) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 0, eta_oo = 0
  // eta = eta_0 * (l * y)^(n-1)
  viscosity = shear_rate_squared * dealii::make_vectorized_array<Number>(
                                     generalized_newtonian_model_data.lambda_squared);
  viscosity =
    std::pow(viscosity,
             dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.n - 1.0));
  viscosity *= dealii::make_vectorized_array<Number>(
    generalized_newtonian_model_data.kinematic_viscosity_upper_limit);
}

template class ViscosityModel<2, float, true, true>;
template class ViscosityModel<2, double, true, true>;
template class ViscosityModel<3, float, true, true>;
template class ViscosityModel<3, double, true, true>;

template class ViscosityModel<2, float, true, false>;
template class ViscosityModel<2, double, true, false>;
template class ViscosityModel<3, float, true, false>;
template class ViscosityModel<3, double, true, false>;

template class ViscosityModel<2, float, false, true>;
template class ViscosityModel<2, double, false, true>;
template class ViscosityModel<3, float, false, true>;
template class ViscosityModel<3, double, false, true>;

} // namespace IncNS
} // namespace ExaDG
