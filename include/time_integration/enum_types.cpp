/*
 * enum_types.cpp
 *
 *  Created on: Apr 1, 2019
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "enum_types.h"

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
