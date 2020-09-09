/*
 * enum_types.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/solvers/enum_types.h>

namespace ExaDG
{
namespace Elementwise
{
using namespace dealii;

std::string
enum_to_string(Solver const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case Solver::Undefined:
      string_type = "Undefined";
      break;
    case Solver::CG:
      string_type = "CG";
      break;
    case Solver::GMRES:
      string_type = "GMRES";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

} // namespace Elementwise
} // namespace ExaDG
