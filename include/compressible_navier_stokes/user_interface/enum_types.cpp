/*
 * enum_types.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "enum_types.h"

using namespace dealii;

namespace CompNS
{
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
enum_to_string(TriangulationType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TriangulationType::Undefined:
      string_type = "Undefined";
      break;
    case TriangulationType::Distributed:
      string_type = "Distributed";
      break;
    case TriangulationType::FullyDistributed:
      string_type = "FullyDistributed";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

} // namespace CompNS
