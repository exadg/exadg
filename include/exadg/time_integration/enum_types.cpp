/*
 * enum_types.cpp
 *
 *  Created on: Apr 1, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/time_integration/enum_types.h>

namespace ExaDG
{
using namespace dealii;

std::string
enum_to_string(CFLConditionType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case CFLConditionType::VelocityNorm:
      string_type = "VelocityNorm";
      break;
    case CFLConditionType::VelocityComponents:
      string_type = "VelocityComponents";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(GenAlphaType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case GenAlphaType::Newmark:
      string_type = "Newmark";
      break;
    case GenAlphaType::GenAlpha:
      string_type = "GenAlpha";
      break;
    case GenAlphaType::HHTAlpha:
      string_type = "HHTAlpha";
      break;
    case GenAlphaType::BossakAlpha:
      string_type = "BossakAlpha";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

} // namespace ExaDG
