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
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

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
      string_type = "FE_Q";
      break;
    case SpatialDiscretization::DG:
      string_type = "FE_DGQ";
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
