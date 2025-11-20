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

#include "viscosity_calculator.h"
#include <exadg/incompressible_flow_with_rans/calculator/viscosity_calculator.h>

namespace ExaDG
{
namespace NSRans
{
template<int dim, typename Number>
ViscosityCalculator<dim, Number>::ViscosityCalculator()
{
}

template<int dim, typename Number>
ViscosityCalculator<dim, Number>::~ViscosityCalculator()
{
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  RANS::TurbulenceModelData const &                      turbulence_model_data_in,
  unsigned int const                                     dof_index_in,
  unsigned int const                                     quad_index_in)
{
  matrix_free = &matrix_free_in;

  turbulence_model_data = turbulence_model_data_in;
  model_coefficients = turbulence_model_data_in.turbulence_data_base->get_all_coefficients();
  
  dof_index = dof_index_in;

  quad_index = quad_index_in;

  matrix_free->initialize_dof_vector(eddy_viscosity, dof_index_in);
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::set_turbulent_kinetic_energy(VectorType const & tke_in)
{
  turbulent_kinetic_energy = &tke_in;
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::set_tke_dissipation_rate(VectorType const & epsilon_in)
{
  tke_dissipation_rate = &epsilon_in;
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number>
ViscosityCalculator<dim, Number>::get_eddy_viscosity() const
{
  return eddy_viscosity;
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::calculate_eddy_viscosity()
{
  VectorType dummy;
  matrix_free->cell_loop(&This::cell_loop_set_viscosity,
                          this,
                          eddy_viscosity,
                          dummy);
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::cell_loop_set_viscosity(dealii::MatrixFree<dim, Number> const & matrix_free,
                                                          VectorType & dst,
                                                          VectorType const &,
                                                          Range const & cell_range) const
{
  IntegratorCell integrator_scalar_1(matrix_free,
                                     dof_index,
                                     quad_index);
  IntegratorCell integrator_scalar_2(matrix_free,
                                     dof_index,
                                     quad_index);
  IntegratorCell integrator_viscosity(matrix_free,
                                      dof_index,
                                      quad_index);
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    if(turbulence_model_data.turbulence_model==RANS::TurbulenceEddyViscosityModel::PrandtlMixingLength)
    {
      integrator_scalar_1.reinit(cell);
      integrator_scalar_1.read_dof_values(*turbulent_kinetic_energy);
    }
    else if(turbulence_model_data.turbulence_model==RANS::TurbulenceEddyViscosityModel::StandardKEpsilon)
    {
      integrator_scalar_1.reinit(cell);
      integrator_scalar_1.read_dof_values(*turbulent_kinetic_energy);

      integrator_scalar_2.reinit(cell);
      integrator_scalar_2.read_dof_values(*tke_dissipation_rate);
    }
    else
  {
      AssertThrow(false, dealii::ExcMessage("RANS::TurbulenceEddyViscosityModel not specified"));
    }

    for(unsigned int dof = 0; dof < integrator_viscosity.dofs_per_cell; ++dof)
    {
      scalar viscosity = dealii::make_vectorized_array<Number>(0.0);
      if(turbulence_model_data.turbulence_model==RANS::TurbulenceEddyViscosityModel::PrandtlMixingLength)
      {
        scalar tke = integrator_scalar_1.get_dof_value(dof);
        add_viscosity(tke, viscosity);
      }
      else if(turbulence_model_data.turbulence_model==RANS::TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        scalar tke = integrator_scalar_1.get_dof_value(dof);
        scalar epsilon = integrator_scalar_2.get_dof_value(dof);
        add_viscosity(tke, epsilon, viscosity);
      }
      else
    {
        AssertThrow(false, dealii::ExcMessage("RANS::TurbulenceEddyViscosityModel not specified"));
      }

      integrator_viscosity.submit_dof_value(viscosity, dof);
    }
    integrator_viscosity.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::add_viscosity(scalar const & scalar_1,
                                                scalar & viscosity) const
{
  switch(turbulence_model_data.turbulence_model)
  {
    case RANS::TurbulenceEddyViscosityModel::PrandtlMixingLength:
      prandtl_mixing_length_model(scalar_1, viscosity);
      break;
    case RANS::TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(false, dealii::ExcMessage("RANS::TurbulenceEddyViscosityModel must be specified"));
      break;
  }
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::add_viscosity(scalar const & scalar_1,
                                                scalar const & scalar_2,
                                                scalar & viscosity) const
{
  switch(turbulence_model_data.turbulence_model)
  {
    case RANS::TurbulenceEddyViscosityModel::StandardKEpsilon:
      standard_k_epsilon_model(scalar_1, scalar_2, viscosity);
      break;
    case RANS::TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(false, dealii::ExcMessage("RANS::TurbulenceEddyViscosityModel must be specified"));
      break;
  }
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::prandtl_mixing_length_model(scalar const & tke,
                                                              scalar & viscosity) const
{
  double length_scale = model_coefficients[2];
  if (turbulence_model_data.positivity_preserving_limiter==RANS::PositivityPreservingLimiter::LogarithmicTransportVariable) {
    viscosity = std::exp(tke / dealii::make_vectorized_array<Number>(2.0)) * length_scale;
  }
  else if (turbulence_model_data.positivity_preserving_limiter==RANS::PositivityPreservingLimiter::Clipper) {
    viscosity = std::pow(tke, dealii::make_vectorized_array<Number>(1.0/2.0)) * length_scale;
  }
  else {
    AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::standard_k_epsilon_model(scalar const & tke,
                                              scalar const & epsilon,
                                              scalar & viscosity) const
{
  double C_mu = model_coefficients[3];

  if (turbulence_model_data.positivity_preserving_limiter==RANS::PositivityPreservingLimiter::LogarithmicTransportVariable) {
    viscosity = C_mu * std::exp((dealii::make_vectorized_array<Number>(2.0) * tke) - epsilon);
  }
  else if (turbulence_model_data.positivity_preserving_limiter==RANS::PositivityPreservingLimiter::Clipper) {
    viscosity = C_mu * std::pow(tke, dealii::make_vectorized_array<Number>(2.0)) / epsilon;
  }
  else {
    AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
}
template class ViscosityCalculator<2, float>;
template class ViscosityCalculator<2, double>;
template class ViscosityCalculator<3, float>;
template class ViscosityCalculator<3, double>;

} // namespace NSRans
} // namespace ExaDG
