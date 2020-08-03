/*
 * enum_types.cpp
 *
 *  Created on: Feb 25, 2020
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "enum_types.h"

using namespace dealii;

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

std::string
enum_to_string(MappingType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MappingType::Affine:
      string_type = "Affine";
      break;
    case MappingType::Quadratic:
      string_type = "Quadratic";
      break;
    case MappingType::Cubic:
      string_type = "Cubic";
      break;
    case MappingType::Isoparametric:
      string_type = "Isoparametric";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}
