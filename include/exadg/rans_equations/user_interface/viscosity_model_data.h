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

#ifndef INCLUDE_EXADG_RANS_EQUATIONS_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_
#define INCLUDE_EXADG_RANS_EQUATIONS_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace RANS
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
  PrandtlMixingLength,
  StandardKEpsilon
};

enum class VaryingViscosityType
{
  CombinedViscosity,
  EddyViscosity
};

enum class PositivityPreservingLimiter
{
  Undefined,
  Clipper,
  LogarithmicTransportVariable
};

struct TurbulenceModelData
{
  TurbulenceModelData()
  {
  }

  TurbulenceEddyViscosityModel turbulence_model{TurbulenceEddyViscosityModel::Undefined};
  bool                         is_active{false};
  bool                         production_term{false};
  bool                         dissipation_term{false};
  double                       turbulent_length_scale{1.0};
  PositivityPreservingLimiter  positivity_preserving_limiter{PositivityPreservingLimiter::Undefined}; 

  void
  check() const
  {
    AssertThrow(is_active, dealii::ExcMessage("Turbulence model is inactive."));
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << std::endl << "Turbulence:" << std::endl;

    print_parameter(pcout, "Use turbulence model", is_active);
    print_parameter(pcout, "Use production term", production_term);
    print_parameter(pcout, "Use dissipation term", dissipation_term);

    if(is_active)
    {
      print_parameter(pcout, "Turbulence model", turbulence_model);
      print_parameter(pcout, "Turbulent length scale", turbulent_length_scale);
      print_parameter(pcout, "Positivity preserving limiter", positivity_preserving_limiter);
    }
  }
};


} // namespace RANSEqns
} // namespace ExaDG

#endif /* INCLUDE_EXADG_RANS_EQUATIONS_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_ */
