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
  {
  }

  TurbulenceEddyViscosityModel turbulence_model{TurbulenceEddyViscosityModel::Undefined};
  bool                         is_active{false};
  double                       constant{0.0}; // model constant

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
  {
  }

  GeneralizedNewtonianViscosityModel generalized_newtonian_model {GeneralizedNewtonianViscosityModel::Undefined};
  bool                               is_active{false};

  // parameters of generalized Carreau-Yasuda model
  double viscosity_margin{0.0};
  double kappa{0.0};
  double lambda{0.0};
  double a{0.0};
  double n{0.0};

  void
  check() const
  {
    AssertThrow(is_active, dealii::ExcMessage("Generalized Newtonian model is inactive."));
    AssertThrow(generalized_newtonian_model != GeneralizedNewtonianViscosityModel::Undefined,
                dealii::ExcMessage("GenerelizedNewtonianViscosityModel not defined."));
  }
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_ */
