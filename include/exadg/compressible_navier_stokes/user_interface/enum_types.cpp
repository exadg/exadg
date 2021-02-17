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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/compressible_navier_stokes/user_interface/enum_types.h>

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;


std::string
enum_to_string(EquationType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case EquationType::Undefined:
      string_type = "Undefined";
      break;
    case EquationType::Euler:
      string_type = "Euler";
      break;
    case EquationType::NavierStokes:
      string_type = "NavierStokes";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(TemporalDiscretization const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TemporalDiscretization::Undefined:
      string_type = "Undefined";
      break;
    case TemporalDiscretization::ExplRK:
      string_type = "ExplRK";
      break;
    case TemporalDiscretization::ExplRK3Stage4Reg2C:
      string_type = "ExplRK3Stage4Reg2C";
      break;
    case TemporalDiscretization::ExplRK3Stage7Reg2:
      string_type = "ExplRK3Stage7Reg2";
      break;
    case TemporalDiscretization::ExplRK4Stage5Reg2C:
      string_type = "ExplRK4Stage5Reg2C";
      break;
    case TemporalDiscretization::ExplRK4Stage8Reg2:
      string_type = "ExplRK4Stage8Reg2";
      break;
    case TemporalDiscretization::ExplRK4Stage5Reg3C:
      string_type = "ExplRK4Stage5Reg3C";
      break;
    case TemporalDiscretization::ExplRK5Stage9Reg2S:
      string_type = "ExplRK5Stage9Reg2S";
      break;
    case TemporalDiscretization::SSPRK:
      string_type = "SSPRK";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


std::string
enum_to_string(TimeStepCalculation const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TimeStepCalculation::Undefined:
      string_type = "Undefined";
      break;
    case TimeStepCalculation::UserSpecified:
      string_type = "UserSpecified";
      break;
    case TimeStepCalculation::CFL:
      string_type = "CFL";
      break;
    case TimeStepCalculation::Diffusion:
      string_type = "Diffusion";
      break;
    case TimeStepCalculation::CFLAndDiffusion:
      string_type = "CFLAndDiffusion";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


std::string
enum_to_string(QuadratureRule const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case QuadratureRule::Standard:
      string_type = "Standard";
      break;
    case QuadratureRule::Overintegration32k:
      string_type = "3/2 k over-integration";
      break;
    case QuadratureRule::Overintegration2k:
      string_type = "2k over-integration";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

} // namespace CompNS
} // namespace ExaDG
