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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <exadg/rans_equations/spatial_discretization/turbulence_model.h>
#include <exadg/operators/quadrature.h>
#include <cmath>
#include <memory>
#include "exadg/rans_equations/user_interface/enum_types.h"
#include "exadg/rans_equations/user_interface/viscosity_model_data.h"

namespace ExaDG
{
namespace RANS
{
template<int dim, typename Number>
TurbulenceModel<dim, Number>::TurbulenceModel()
{
}

template<int dim, typename Number>
TurbulenceModel<dim, Number>::~TurbulenceModel()
{
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  TurbulenceModelData const &                            turbulence_model_data_in,
  unsigned int const                                     dof_index_in,
  unsigned int const                                     quad_index_in)
{
  Base::initialize(matrix_free_in, dof_index_in, quad_index_in);

  turbulence_model_data = turbulence_model_data_in;

  turbulence_model_data.check();

  viscosity_coefficients.initialize(matrix_free_in, quad_index_in, true, true);
  viscosity_coefficients.set_coefficients(diffusivity);
  eddy_viscosity_coefficients.initialize(matrix_free_in, quad_index_in, true, true);
  eddy_viscosity_coefficients.set_coefficients(1.0);

  this->matrix_free->initialize_dof_vector(eddy_viscosity, dof_index_in);

  model_coefficients = this->turbulence_data_base->get_all_coefficients();
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_viscosity(VectorType const & solution) 
{
  this->set_constant_coefficient(this->diffusivity);

  this->add_viscosity(solution);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::add_viscosity(VectorType const & solution) 
{
  VectorType dummy;

  this->matrix_free->loop(&This::cell_loop_set_coefficients,
                          &This::face_loop_set_coefficients,
                          &This::boundary_face_loop_set_coefficients,
                          this,
                          dummy,
                          solution);
  /*this->matrix_free->cell_loop(&This::cell_loop_set_coefficients,*/
  /*                             this,*/
  /*                             dummy,*/
  /*                             solution);*/
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      cell_range) 
{
  IntegratorCell integrator(matrix_free,
                             this->dof_index,
                             this->quad_index);

  IntegratorCell integrator_secondary(matrix_free,
                                      this->dof_index,
                                      this->quad_index);

  IntegratorCell integrator_viscosity(matrix_free,
                             this->dof_index,
                             this->quad_index);

  /*std::string file_tke = "/home/bharan/computation/tutorials/exadg/incompressible_flow_with_rans/channel_flow/output/tke.csv";*/
  /*std::string file_epsilon = "/home/bharan/computation/tutorials/exadg/incompressible_flow_with_rans/channel_flow/output/epsilon.csv";*/
  /*std::string file_nuT = "/home/bharan/computation/tutorials/exadg/incompressible_flow_with_rans/channel_flow/output/nuT.csv";*/
  // loop over all cells
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    integrator_viscosity.reinit(cell);
    integrator_viscosity.read_dof_values(eddy_viscosity);

    // we only need the values
    integrator.evaluate(dealii::EvaluationFlags::values);

    if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
      integrator_secondary.reinit(cell);
      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        integrator_secondary.read_dof_values(*tke_dissipation_rate);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        integrator_secondary.read_dof_values(*turbulent_kinetic_energy);
      }
      integrator_secondary.evaluate(dealii::EvaluationFlags::values);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      scalar solution_value = integrator.get_value(q);

      // get the current viscosity
      scalar current_viscosity = this->diffusivity;
      scalar viscosity = 1.0;

        /*dealii::Point<dim, scalar> pnt = integrator.quadrature_point(q);*/

      if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLength) {
        add_one_equation_turbulent_viscosity(viscosity,
                                             solution_value);
      }
      else if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
        scalar secondary_variable = integrator_secondary.get_value(q);
        if (scalar_type==ScalarType::TurbulentKineticEnergy) {
          add_two_equation_turbulent_viscosity(viscosity, solution_value, secondary_variable);          
          /*SaveToCSV<dim, Number> csv_tke(file_tke, solution_value, pnt);*/
        }
        else if (scalar_type==ScalarType::TKEDissipationRate) {
          add_two_equation_turbulent_viscosity(viscosity, secondary_variable, solution_value);          
          /*SaveToCSV<dim, Number> csv_tke(file_epsilon, solution_value, pnt);*/
        }
      }

      eddy_viscosity_coefficients.set_coefficient_cell(cell, q, viscosity);
      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        viscosity = current_viscosity + (viscosity / model_coefficients[0]);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        viscosity = current_viscosity + (viscosity / model_coefficients[4]);
      }
      // set the coefficients
      viscosity_coefficients.set_coefficient_cell(cell, q, viscosity);
    }

    for (uint dof = 0; dof < integrator_viscosity.dofs_per_cell; ++dof){
      // calculate velocity gradient
      scalar solution_value = integrator.get_dof_value(dof);

      scalar viscosity = 0.0;

      if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLength) {
        add_one_equation_turbulent_viscosity(viscosity,
                                             solution_value);
      }
      else if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
        scalar secondary_variable = integrator_secondary.get_dof_value(dof);
        if (scalar_type==ScalarType::TurbulentKineticEnergy) {
          add_two_equation_turbulent_viscosity(viscosity, solution_value, secondary_variable);          
        }
        else if (scalar_type==ScalarType::TKEDissipationRate) {
          add_two_equation_turbulent_viscosity(viscosity, secondary_variable, solution_value);          
        }
      }
      integrator_viscosity.submit_dof_value(viscosity, dof);
    }
    integrator_viscosity.distribute_local_to_global(eddy_viscosity);
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) 
{
  IntegratorFace integrator_m(matrix_free,
                               true,
                               this->dof_index,
                               this->quad_index);
  IntegratorFace integrator_p(matrix_free,
                               false,
                               this->dof_index,
                               this->quad_index);
  IntegratorFace integrator_secondary_m(matrix_free,
                               true,
                               this->dof_index,
                               this->quad_index);
  IntegratorFace integrator_secondary_p(matrix_free,
                               false,
                               this->dof_index,
                               this->quad_index);

  // loop over all interior faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.read_dof_values(src);
    integrator_p.read_dof_values(src);

    // we only need the gradient
    integrator_m.evaluate(dealii::EvaluationFlags::values);
    integrator_p.evaluate(dealii::EvaluationFlags::values);

    if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
      integrator_secondary_m.reinit(face);
      integrator_secondary_p.reinit(face);
      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        integrator_secondary_m.read_dof_values(*tke_dissipation_rate);
        integrator_secondary_p.read_dof_values(*tke_dissipation_rate);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        integrator_secondary_m.read_dof_values(*turbulent_kinetic_energy);
        integrator_secondary_p.read_dof_values(*turbulent_kinetic_energy);
      }
      integrator_secondary_m.evaluate(dealii::EvaluationFlags::values);
      integrator_secondary_p.evaluate(dealii::EvaluationFlags::values);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      // calculate velocity gradient for both elements adjacent to the current face
      scalar solution_value          = integrator_m.get_value(q);
      scalar solution_value_neighbor = integrator_p.get_value(q);

      // get the coefficients
      scalar current_viscosity          = this->diffusivity;
      scalar current_viscosity_neighbor = this->diffusivity;

      scalar viscosity = 1.0;
      scalar viscosity_neighbor = 1.0;

      if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLength) {
        add_one_equation_turbulent_viscosity(viscosity,
                                             solution_value);
        add_one_equation_turbulent_viscosity(viscosity_neighbor,
                                             solution_value_neighbor);
      }
      else if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
        scalar secondary_variable = integrator_secondary_m.get_value(q);
        scalar secondary_variable_neighbor = integrator_secondary_p.get_value(q);
        if (scalar_type==ScalarType::TurbulentKineticEnergy) {
          add_two_equation_turbulent_viscosity(viscosity, solution_value, secondary_variable);          
          add_two_equation_turbulent_viscosity(viscosity, solution_value_neighbor, secondary_variable_neighbor);          
        }
        else if (scalar_type==ScalarType::TKEDissipationRate) {
          add_two_equation_turbulent_viscosity(viscosity, secondary_variable, solution_value);
          add_two_equation_turbulent_viscosity(viscosity, secondary_variable_neighbor, solution_value_neighbor);
        }
      }

      eddy_viscosity_coefficients.set_coefficient_face(face, q, viscosity);
      eddy_viscosity_coefficients.set_coefficient_face_neighbor(face, q, viscosity_neighbor);

      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        viscosity = current_viscosity + (viscosity / model_coefficients[0]);
        viscosity_neighbor = current_viscosity_neighbor + (viscosity_neighbor / model_coefficients[0]);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        viscosity = current_viscosity + (viscosity / model_coefficients[4]);
        viscosity_neighbor = current_viscosity_neighbor + (viscosity_neighbor / model_coefficients[4]);
      }
      // set the coefficients
      viscosity_coefficients.set_coefficient_face(face, q, viscosity);
      viscosity_coefficients.set_coefficient_face_neighbor(face, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::boundary_face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) 
{
  IntegratorFace integrator(matrix_free,
                             true,
                             this->dof_index,
                             this->quad_index);
  IntegratorFace integrator_secondary(matrix_free,
                             true,
                             this->dof_index,
                             this->quad_index);

  // loop over all boundary faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::values);

    if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
      integrator_secondary.reinit(face);
      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        integrator_secondary.read_dof_values(*tke_dissipation_rate);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        integrator_secondary.read_dof_values(*turbulent_kinetic_energy);
      }
      integrator_secondary.evaluate(dealii::EvaluationFlags::values);
    }

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      scalar solution_value = integrator.get_value(q);

      // get the coefficients
      scalar current_viscosity = this->diffusivity;
      scalar viscosity = 1.0;

      if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::PrandtlMixingLength) {
        add_one_equation_turbulent_viscosity(viscosity,
                                             solution_value);
      }
      else if (turbulence_model_data.turbulence_model==TurbulenceEddyViscosityModel::StandardKEpsilon) {
        scalar secondary_variable = integrator_secondary.get_value(q);
        if (scalar_type==ScalarType::TurbulentKineticEnergy) {
          add_two_equation_turbulent_viscosity(viscosity, solution_value, secondary_variable);
        }
        else if (scalar_type==ScalarType::TKEDissipationRate) {
          add_two_equation_turbulent_viscosity(viscosity, secondary_variable, solution_value);
        }
      }

      eddy_viscosity_coefficients.set_coefficient_face(face, q, viscosity);

      viscosity = current_viscosity + (viscosity / turbulence_data_base->sigma_k);

      // set the coefficients
      viscosity_coefficients.set_coefficient_face(face, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::add_one_equation_turbulent_viscosity(scalar &       viscosity,
                                                      scalar const & solution) const
{
  switch(turbulence_model_data.turbulence_model)
  {
    case TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(turbulence_model_data.turbulence_model != TurbulenceEddyViscosityModel::Undefined,
                  dealii::ExcMessage("Parameter must be defined."));
      break;
    case TurbulenceEddyViscosityModel::PrandtlMixingLength:
      prandtl_mixing_length_model(solution, viscosity);
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage("This TurbulenceEddyViscosityModel is not implemented."));
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::add_two_equation_turbulent_viscosity(scalar &       viscosity,
                                                                   scalar const & tke,
                                                                   scalar const & epsilon) const
{
  switch(turbulence_model_data.turbulence_model)
  {
    case TurbulenceEddyViscosityModel::Undefined:
      AssertThrow(turbulence_model_data.turbulence_model != TurbulenceEddyViscosityModel::Undefined,
                  dealii::ExcMessage("Parameter must be defined."));
      break;
    case TurbulenceEddyViscosityModel::StandardKEpsilon:
      k_epsilon_model(tke, epsilon, viscosity);
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage("This TurbulenceEddyViscosityModel is not implemented."));
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_turbulent_kinetic_energy(VectorType const & tke_in)
{
  this->turbulent_kinetic_energy = &tke_in;
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_tke_dissipation_rate(VectorType const & epsilon_in)
{
  this->tke_dissipation_rate = &epsilon_in;
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::get_eddy_viscosity(VectorType & dst)
{
  dst.equ(1.0, this->eddy_viscosity);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::prandtl_mixing_length_model(scalar const & sol,
                                                scalar &       viscosity) const
{
  double length_scale = model_coefficients[2];
  if (turbulence_model_data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
    viscosity = std::exp(sol / 2.0) * length_scale;
  }
  else if (turbulence_model_data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper) {
    viscosity = std::pow(sol, dealii::make_vectorized_array<Number>(1.0/2.0)) * length_scale;
  }
  else {
    AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::k_epsilon_model(scalar const & tke,
                                              scalar const & epsilon,
                                              scalar & viscosity) const
{
  double C_mu = model_coefficients[3];

  if (turbulence_model_data.positivity_preserving_limiter==PositivityPreservingLimiter::LogarithmicTransportVariable) {
    viscosity = C_mu * std::exp((dealii::make_vectorized_array<Number>(2.0) *tke) - epsilon);
  }
  else if (turbulence_model_data.positivity_preserving_limiter==PositivityPreservingLimiter::Clipper) {
    viscosity = C_mu * std::pow(tke, dealii::make_vectorized_array<Number>(2.0)) / epsilon;
  }
  else {
    AssertThrow(false, dealii::ExcMessage("PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
  /*std::cout << "tke : " << tke << std::endl;*/
  /*std::cout << "epsilon : " << epsilon << std::endl;*/
  /*std::cout << "eddy viscosity : " << viscosity << std::endl;*/
}

template<int dim, typename Number>
std::shared_ptr<TurbulenceDataBase> 
TurbulenceModel<dim, Number>::create_turbulence_data()
{
  switch(turbulence_model_data.turbulence_model)
  {
    case TurbulenceEddyViscosityModel::PrandtlMixingLength:
      return std::make_shared<PrandtlMixingLengthData>();
      break;
    case TurbulenceEddyViscosityModel::StandardKEpsilon:
      return std::make_shared<StandardKEpsilonData>();
      break;
  }
}

template class TurbulenceModel<2, float>;
template class TurbulenceModel<2, double>;
template class TurbulenceModel<3, float>;
template class TurbulenceModel<3, double>;

} // namespace RANSEqns
} // namespace ExaDG
