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

#include <exadg/incompressible_navier_stokes/spatial_discretization/generalized_newtonian_model.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
GeneralizedNewtonianModel<dim, Number>::GeneralizedNewtonianModel() : Base()
{
}

template<int dim, typename Number>
GeneralizedNewtonianModel<dim, Number>::~GeneralizedNewtonianModel()
{
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
  GeneralizedNewtonianModelData const &                  generalized_newtonian_model_data_in,
  unsigned int                                           dof_index_velocity_in,
  unsigned int                                           quad_index_velocity_linear_in)
{
  Base::initialize(matrix_free_in,
                   viscous_kernel_in,
                   dof_index_velocity_in,
                   quad_index_velocity_linear_in);

  generalized_newtonian_model_data = generalized_newtonian_model_data_in;
  generalized_newtonian_model_data.check();
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::set_viscosity(VectorType const & velocity) const
{
  Base::viscous_kernel->set_constant_coefficient(Base::viscosity_newtonian_limit);

  this->add_viscosity(velocity);
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::add_viscosity(VectorType const & velocity) const
{
  VectorType dummy;

  Base::matrix_free->loop(&This::cell_loop_set_coefficients,
                          &This::face_loop_set_coefficients,
                          &This::boundary_face_loop_set_coefficients,
                          this,
                          dummy,
                          velocity);
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      cell_range) const
{
  CellIntegratorU integrator(matrix_free,
                             Base::dof_index_velocity,
                             Base::quad_index_velocity_linear);

  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      tensor velocity_gradient = integrator.get_gradient(q);

      // get the current viscosity
      scalar viscosity = Base::viscous_kernel->get_viscosity_cell(cell, q);

      add_generalized_newtonian_viscosity(viscosity, velocity_gradient);

      // set the coefficients
      Base::viscous_kernel->set_coefficient_cell(cell, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) const
{
  FaceIntegratorU integrator_m(matrix_free,
                               true,
                               Base::dof_index_velocity,
                               Base::quad_index_velocity_linear);
  FaceIntegratorU integrator_p(matrix_free,
                               false,
                               Base::dof_index_velocity,
                               Base::quad_index_velocity_linear);

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

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      // calculate velocity gradient for both elements adjacent to the current face
      tensor velocity_gradient          = integrator_m.get_gradient(q);
      tensor velocity_gradient_neighbor = integrator_p.get_gradient(q);

      // get the coefficients
      scalar viscosity          = Base::viscous_kernel->get_coefficient_face(face, q);
      scalar viscosity_neighbor = Base::viscous_kernel->get_coefficient_face_neighbor(face, q);

      add_generalized_newtonian_viscosity(viscosity, velocity_gradient);
      add_generalized_newtonian_viscosity(viscosity_neighbor, velocity_gradient_neighbor);

      // set the coefficients
      Base::viscous_kernel->set_coefficient_face(face, q, viscosity);
      Base::viscous_kernel->set_coefficient_face_neighbor(face, q, viscosity_neighbor);
    }
  }
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::boundary_face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) const
{
  FaceIntegratorU integrator(matrix_free,
                             true,
                             Base::dof_index_velocity,
                             Base::quad_index_velocity_linear);

  // loop over all boundary faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      tensor velocity_gradient = integrator.get_gradient(q);

      // get the coefficients
      scalar viscosity = Base::viscous_kernel->get_coefficient_face(face, q);

      add_generalized_newtonian_viscosity(viscosity, velocity_gradient);

      // set the coefficients
      Base::viscous_kernel->set_coefficient_face(face, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::add_generalized_newtonian_viscosity(
  scalar &       viscosity,
  tensor const & velocity_gradient) const
{

  tensor symmetric_velocity_gradient = 0.5 * (velocity_gradient + transpose(velocity_gradient));

  scalar shear_rate = std::sqrt(2.0 * scalar_product(symmetric_velocity_gradient, symmetric_velocity_gradient));

  scalar viscosity_factor;
  compute_viscosity_factor(viscosity_factor, shear_rate);

  viscosity += viscosity_factor * generalized_newtonian_model_data.viscosity_margin;
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::compute_viscosity_factor(scalar &       viscosity_factor,
                                                                 scalar const & shear_rate) const
{
  switch(generalized_newtonian_model_data.generalized_newtonian_model)
  {
    case GeneralizedNewtonianViscosityModel::Undefined:
      AssertThrow(generalized_newtonian_model_data.generalized_newtonian_model !=
                    GeneralizedNewtonianViscosityModel::Undefined,
                  dealii::ExcMessage("parameter must be defined"));
      break;
    case GeneralizedNewtonianViscosityModel::GeneralizedCarreauYasuda:
      generalized_carreau_yasuda_generalized_newtonian_model(viscosity_factor, shear_rate);
      break;
    case GeneralizedNewtonianViscosityModel::Carreau:
      carreau_generalized_newtonian_model(viscosity_factor, shear_rate);
      break;
    case GeneralizedNewtonianViscosityModel::Cross:
      cross_generalized_newtonian_model(viscosity_factor, shear_rate);
      break;
    case GeneralizedNewtonianViscosityModel::SimplifiedCross:
      simplified_cross_generalized_newtonian_model(viscosity_factor, shear_rate);
      break;
    case GeneralizedNewtonianViscosityModel::PowerLaw:
      power_law_generalized_newtonian_model(viscosity_factor, shear_rate);
      break;
    default:
      AssertThrow(
        false, dealii::ExcMessage("This GeneralizedNewtonianViscosityModel is not implemented."));
  }
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::generalized_carreau_yasuda_generalized_newtonian_model(
  scalar &       viscosity_factor,
  scalar const & shear_rate) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  viscosity_factor =
    shear_rate * dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.lambda);
  viscosity_factor =
    std::pow(viscosity_factor,
             dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.a));
  viscosity_factor += dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.kappa);
  viscosity_factor =
    std::pow(viscosity_factor,
             dealii::make_vectorized_array<Number>((generalized_newtonian_model_data.n - 1.0) /
                                                   generalized_newtonian_model_data.a));
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::carreau_generalized_newtonian_model(
  scalar &       viscosity_factor,
  scalar const & shear_rate) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 1 and a = 2
  // eta = eta_oo + (eta_0 - eta_oo) * [1 + l^2 * y^2]^[(n-1)/2]
  viscosity_factor = shear_rate * shear_rate *
                     dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.lambda *
                                                           generalized_newtonian_model_data.lambda);

  viscosity_factor += dealii::make_vectorized_array<Number>(1.0);

  viscosity_factor = std::pow(viscosity_factor,
                              dealii::make_vectorized_array<Number>(
                                (generalized_newtonian_model_data.n - 1.0) / 2.0));
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::cross_generalized_newtonian_model(
  scalar &       viscosity_factor,
  scalar const & shear_rate) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 1 and n = 1 - a
  // eta = eta_oo + (eta_0 - eta_oo) * [1 + (l * y)^a]^(-1)
  viscosity_factor =
    shear_rate * dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.lambda);

  viscosity_factor =
    std::pow(viscosity_factor,
             dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.a));

  viscosity_factor += dealii::make_vectorized_array<Number>(1.0);
  viscosity_factor = dealii::make_vectorized_array<Number>(1.0) / viscosity_factor;
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::simplified_cross_generalized_newtonian_model(
  scalar &       viscosity_factor,
  scalar const & shear_rate) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 1, a = 1 and n = 0
  // eta = eta_oo + (eta_0 - eta_oo) * [1 + l * y]^(-1)
  viscosity_factor =
    shear_rate * dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.lambda);

  viscosity_factor += dealii::make_vectorized_array<Number>(1.0);

  viscosity_factor = dealii::make_vectorized_array<Number>(1.0) / viscosity_factor;
}

template<int dim, typename Number>
void
GeneralizedNewtonianModel<dim, Number>::power_law_generalized_newtonian_model(
  scalar &       viscosity_factor,
  scalar const & shear_rate) const
{
  // eta = eta_oo + (eta_0 - eta_oo) * [k + (l * y)^a]^[(n-1)/a]
  // with k = 0
  // eta = eta_oo + (eta_0 - eta_oo) * (l * y)^(n-1)
  viscosity_factor =
    shear_rate * dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.lambda);

  viscosity_factor =
    std::pow(viscosity_factor,
             dealii::make_vectorized_array<Number>(generalized_newtonian_model_data.n - 1.0));
}

template class GeneralizedNewtonianModel<2, float>;
template class GeneralizedNewtonianModel<2, double>;
template class GeneralizedNewtonianModel<3, float>;
template class GeneralizedNewtonianModel<3, double>;

} // namespace IncNS
} // namespace ExaDG
