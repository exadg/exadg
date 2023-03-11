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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_

namespace ExaDG
{
namespace IncNS
{
/*
 *  Turbulence model data.
 */
struct TurbulenceModelData
{
  TurbulenceModelData()
    : turbulence_model(TurbulenceEddyViscosityModel::Undefined), constant(0.0), viscosity(0.0)
  {
  }

  TurbulenceEddyViscosityModel turbulence_model;

  // model constant and the fluid's kinematic (physical) viscosity
  double constant;
  double viscosity;
};

/*
 * Generalized Newtonian model data.
 */
struct GeneralizedNewtonianModelData
{
  GeneralizedNewtonianModelData()
    : generalized_newtonian_model(GeneralizedNewtonianModel::Undefined),
      viscosity_lower_limit(0.0),
      viscosity_upper_limit(0.0),
      kappa(0.0),
      lambda(0.0),
      a(0.0),
      n(0.0)
  {
  }

  GeneralizedNewtonianModel generalized_newtonian_model;

  // parameters in generalized Carreau-Yasuda model
  double viscosity_lower_limit;
  double viscosity_upper_limit;
  double kappa;
  double lambda;
  double a;
  double n;
};

/*
 * Variable viscosity model data.
 */
struct ViscosityModelData
{
  ViscosityModelData()
    : treatment_of_nonlinear_viscosity(TreatmentOfNonlinearViscosity::Undefined),
      use_turbulence_model(false),
      use_generalized_newtonian_model(false)
  {
  }
  TurbulenceModelData           turbulence_model_data;
  GeneralizedNewtonianModelData generalized_newtonian_model_data;

  TreatmentOfNonlinearViscosity treatment_of_nonlinear_viscosity;
  bool                          use_turbulence_model;
  bool                          use_generalized_newtonian_model;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_ */
