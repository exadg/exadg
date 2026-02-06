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
#include <exadg/incompressible_flow_with_rans/spatial_discretization/viscosity_calculator.h>

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
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  RANS::TurbulenceModelData const &       turbulence_model_data_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in)
{
  matrix_free = &matrix_free_in;

  turbulence_model_data = turbulence_model_data_in;
  model_coefficients    = turbulence_model_data_in.turbulence_data_base->get_all_coefficients();

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
dealii::LinearAlgebra::distributed::Vector<Number> const &
ViscosityCalculator<dim, Number>::get_eddy_viscosity() const
{
  return eddy_viscosity;
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::calculate_eddy_viscosity()
{
  VectorType dummy;
  eddy_viscosity.zero_out_ghost_values();
  matrix_free->cell_loop(&This::cell_loop_set_viscosity, this, eddy_viscosity, dummy);
  eddy_viscosity.update_ghost_values();
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::cell_loop_set_viscosity(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & cell_range) const
{
  IntegratorCell integrator_scalar_1(matrix_free, dof_index, quad_index);
  IntegratorCell integrator_scalar_2(matrix_free, dof_index, quad_index);
  IntegratorCell integrator_viscosity(matrix_free, dof_index, quad_index);
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_viscosity.reinit(cell);
    if(turbulence_model_data.turbulence_model ==
            RANS::TurbulenceEddyViscosityModel::StandardKEpsilon)
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
      if(turbulence_model_data.turbulence_model ==
              RANS::TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        scalar tke     = integrator_scalar_1.get_dof_value(dof);
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
                                                scalar const & scalar_2,
                                                scalar &       viscosity) const
{
  switch(turbulence_model_data.turbulence_model)
  {
    case RANS::TurbulenceEddyViscosityModel::StandardKEpsilon:
      standard_k_epsilon_model(scalar_1, scalar_2, viscosity);
      break;
    case RANS::TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(false,
                  dealii::ExcMessage("RANS::TurbulenceEddyViscosityModel must be specified"));
      break;
  }
}

/*
* \f[  \nu_t = C_{\mu} \frac{k^2}{\epsilon}  \f]
*/
template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::standard_k_epsilon_model(scalar const & tke,
                                                           scalar const & epsilon,
                                                           scalar &       viscosity) const
{
  double C_mu = model_coefficients[3];

  if(turbulence_model_data.positivity_preserving_limiter ==
     RANS::PositivityPreservingLimiter::LogarithmicTransportVariable)
  {
    scalar log_terms = (dealii::make_vectorized_array<Number>(2.0) * tke) - epsilon;
    scalar safe_log_term = std::min(log_terms,
                                        dealii::make_vectorized_array<Number>(10.0));
    scalar calculated_viscosity = C_mu * std::exp(safe_log_term);

    viscosity = std::min(calculated_viscosity,
                                dealii::make_vectorized_array<Number>(100.0));
  }
  else if(turbulence_model_data.positivity_preserving_limiter ==
          RANS::PositivityPreservingLimiter::Clipper)
  {
    viscosity = C_mu * std::pow(tke, dealii::make_vectorized_array<Number>(2.0)) / epsilon;
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::extrapolate_eddy_viscosity_to_dof(
  VectorType &     dst,
  unsigned const & target_dof_index) const
{
  if(target_dof_index == dof_index)
  {
    dst = eddy_viscosity;
    dst.update_ghost_values();
    return;
  }

  extrapolate_to_new_dof(eddy_viscosity, dst, target_dof_index);
}

template<int dim, typename Number>
void
ViscosityCalculator<dim, Number>::extrapolate_to_new_dof(
  VectorType const &   src,
  VectorType &         dst,
  unsigned int const & target_dof_index) const
{
  // 1. Get the DoFHandlers for source and destination
  auto const & dof_handler_src = matrix_free->get_dof_handler(dof_index);
  auto const & dof_handler_dst = matrix_free->get_dof_handler(target_dof_index);

  matrix_free->initialize_dof_vector(dst, target_dof_index);

  // 3. Ensure source vector has ghost values updated
  VectorType src_with_ghosts = src;
  src_with_ghosts.update_ghost_values();

  // 4. Create a continuous function wrapper around the source vector
  dealii::Functions::FEFieldFunction<dim, dealii::LinearAlgebra::distributed::Vector<Number>>
    src_function(dof_handler_src, src_with_ghosts);

  // 5. Interpolate that function onto the destination space
  dealii::VectorTools::interpolate(dof_handler_dst, src_function, dst);

  // 6. Update ghosts so the new vector is ready for use
  dst.update_ghost_values();
}

template class ViscosityCalculator<2, float>;
template class ViscosityCalculator<2, double>;
template class ViscosityCalculator<3, float>;
template class ViscosityCalculator<3, double>;

} // namespace NSRans
} // namespace ExaDG
