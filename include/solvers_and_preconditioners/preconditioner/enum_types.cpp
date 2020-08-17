/*
 * enum_types.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: fehn
 */

#include "enum_types.h"

#include <deal.II/base/exceptions.h>

namespace ExaDG
{
namespace Elementwise
{
using namespace dealii;

std::string
enum_to_string(Preconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case Preconditioner::Undefined:
      string_type = "Undefined";
      break;
    case Preconditioner::None:
      string_type = "None";
      break;
    case Preconditioner::InverseMassMatrix:
      string_type = "InverseMassMatrix";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

} // namespace Elementwise
} // namespace ExaDG
