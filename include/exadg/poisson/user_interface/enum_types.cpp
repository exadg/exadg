/*
 * enum_types.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/poisson/user_interface/enum_types.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

/**************************************************************************************/
/*                                                                                    */
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

std::string
enum_to_string(SpatialDiscretization const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SpatialDiscretization::Undefined:
      string_type = "Undefined";
      break;
    case SpatialDiscretization::CG:
      string_type = "CG";
      break;
    case SpatialDiscretization::DG:
      string_type = "DG";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

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
    case Solver::FGMRES:
      string_type = "FGMRES";
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
} // namespace ExaDG
