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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
/**
 *  Algebraic subgrid-scale turbulence models for LES
 *
 *  Standard constants according to literature:
 *    Smagorinsky: 0.165
 *    Vreman: 0.28
 *    WALE: 0.50
 *    Sigma: 1.35
 */
enum class TurbulenceEddyViscosityModel
{
  Undefined,
  Smagorinsky,
  Vreman,
  WALE,
  Sigma
};

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

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << std::endl << "Turbulence:" << std::endl;

    print_parameter(pcout, "Use turbulence model", is_active);

    if(is_active)
    {
      print_parameter(pcout, "Turbulence model", turbulence_model);
      print_parameter(pcout, "Turbulence model constant", constant);
    }
  }
};

/**
 *  Generalized Newtonian models
 */
enum class GeneralizedNewtonianViscosityModel
{
  Undefined,
  GeneralizedCarreauYasuda
};

struct GeneralizedNewtonianModelData
{
  GeneralizedNewtonianModelData()
  {
  }

  GeneralizedNewtonianViscosityModel generalized_newtonian_model{
    GeneralizedNewtonianViscosityModel::Undefined};
  bool is_active{false};

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

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << std::endl << "Rheology:" << std::endl;

    print_parameter(pcout, "Use generalized Newtonian model", is_active);

    if(is_active)
    {
      print_parameter(pcout, "Generalized Newtonian model", generalized_newtonian_model);
      print_parameter(pcout, "viscosity margin", viscosity_margin);
      print_parameter(pcout, "parameter kappa", kappa);
      print_parameter(pcout, "parameter lambda", lambda);
      print_parameter(pcout, "parameter a", a);
      print_parameter(pcout, "parameter n", n);
    }
  }
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_ */
