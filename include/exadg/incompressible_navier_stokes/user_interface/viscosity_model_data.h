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
    : turbulence_model(TurbulenceEddyViscosityModel::Undefined), is_active(false), constant(0.0)
  {
  }

  TurbulenceEddyViscosityModel turbulence_model;
  bool                         is_active;
  double                       constant; // model constant

  void
  check() const
  {
    AssertThrow(is_active, dealii::ExcMessage("Turbulence model is inactive."));
    AssertThrow(constant > 1e-20, dealii::ExcMessage("Parameter must be greater than zero."));
  }
};

/*
 * Generalized Newtonian model data.
 */
struct GeneralizedNewtonianModelData
{
  GeneralizedNewtonianModelData()
    : generalized_newtonian_model(GeneralizedNewtonianViscosityModel::Undefined),
      is_active(false),
      viscosity_margin(0.0),
      kappa(0.0),
      lambda(0.0),
      a(0.0),
      n(0.0)
  {
  }

  GeneralizedNewtonianViscosityModel generalized_newtonian_model;
  bool                               is_active;

  // parameters of generalized Carreau-Yasuda model
  double viscosity_margin;
  double kappa;
  double lambda;
  double a;
  double n;

  void
  check() const
  {
    AssertThrow(is_active, dealii::ExcMessage("Generalized Newtonian model is inactive."));

    // check assumptions of rheological models and enforce user to set the parameters
    // accordingly in the generalized Carreau-Yasuda model
    if(generalized_newtonian_model == GeneralizedNewtonianViscosityModel::Carreau ||
       generalized_newtonian_model == GeneralizedNewtonianViscosityModel::Cross ||
       generalized_newtonian_model == GeneralizedNewtonianViscosityModel::SimplifiedCross)
    {
      AssertThrow(std::abs(kappa - 1.0) < 1e-20,
                  dealii::ExcMessage("generalized_newtonian_kappa == 1 required for "
                                     "this GeneralizedNewtonianViscosityModel."));
    }

    if(generalized_newtonian_model == GeneralizedNewtonianViscosityModel::Carreau)
    {
      AssertThrow(std::abs(a - 2.0) < 1e-20,
                  dealii::ExcMessage("generalized_newtonian_a == 2 required for"
                                     "GeneralizedNewtonianViscosityModel::Carreau."));
    }

    if(generalized_newtonian_model == GeneralizedNewtonianViscosityModel::Cross)
    {
      AssertThrow(std::abs(n - 1.0 + a) < 1e-20,
                  dealii::ExcMessage("generalized_newtonian_n - 1 == generalized_newtonian_a "
                                     "required for GeneralizedNewtonianViscosityModel::Cross."));
    }

    if(generalized_newtonian_model == GeneralizedNewtonianViscosityModel::SimplifiedCross)
    {
      AssertThrow(std::abs(a - 1.0) < 1e-20,
                  dealii::ExcMessage(
                    "generalized_newtonian_a == 1 "
                    "required for GeneralizedNewtonianViscosityModel::SimplifiedCross."));
      AssertThrow(std::abs(n) < 1e-20,
                  dealii::ExcMessage(
                    "generalized_newtonian_n == 0 "
                    "required for GeneralizedNewtonianViscosityModel::SimplifiedCross."));
    }

    if(generalized_newtonian_model == GeneralizedNewtonianViscosityModel::PowerLaw)
    {
      AssertThrow(std::abs(kappa) < 1e-20,
                  dealii::ExcMessage("generalized_newtonian_kappa == 0 required for "
                                     "required for GeneralizedNewtonianViscosityModel::PowerLaw."));
    }
  }
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_ */
