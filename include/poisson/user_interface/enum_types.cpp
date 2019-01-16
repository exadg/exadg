/*
 * enum_types.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "enum_types.h"

using namespace dealii;

namespace Poisson
{
/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/

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
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

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
    case Preconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case Preconditioner::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case Preconditioner::Multigrid:
      string_type = "Multigrid";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


} // namespace Poisson
