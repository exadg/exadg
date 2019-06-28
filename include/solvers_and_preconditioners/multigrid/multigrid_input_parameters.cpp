/*
 * multigrid_input_parameters.cpp
 *
 *  Created on: Feb 1, 2019
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "multigrid_input_parameters.h"

using namespace dealii;


std::string
enum_to_string(MultigridType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridType::Undefined:
      string_type = "Undefined";
      break;
    case MultigridType::hMG:
      string_type = "h-MG";
      break;
    case MultigridType::pMG:
      string_type = "p-MG";
      break;
    case MultigridType::hpMG:
      string_type = "hp-MG";
      break;
    case MultigridType::phMG:
      string_type = "ph-MG";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PSequenceType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PSequenceType::GoToOne:
      string_type = "GoToOne";
      break;
    case PSequenceType::DecreaseByOne:
      string_type = "DecreaseByOne";
      break;
    case PSequenceType::Bisect:
      string_type = "Bisect";
      break;
    case PSequenceType::Manual:
      string_type = "Manual";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(DG_To_CG_Transfer const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case DG_To_CG_Transfer::None:
      string_type = "None";
      break;
    case DG_To_CG_Transfer::Fine:
      string_type = "Fine";
      break;
    case DG_To_CG_Transfer::Coarse:
      string_type = "Coarse";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MultigridSmoother const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridSmoother::Chebyshev:
      string_type = "Chebyshev";
      break;
    case MultigridSmoother::GMRES:
      string_type = "GMRES";
      break;
    case MultigridSmoother::CG:
      string_type = "CG";
      break;
    case MultigridSmoother::Jacobi:
      string_type = "Jacobi";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MultigridCoarseGridSolver const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridCoarseGridSolver::Chebyshev:
      string_type = "Chebyshev";
      break;
    case MultigridCoarseGridSolver::CG:
      string_type = "CG";
      break;
    case MultigridCoarseGridSolver::GMRES:
      string_type = "GMRES";
      break;
    case MultigridCoarseGridSolver::AMG:
      string_type = "AMG";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


std::string
enum_to_string(MultigridCoarseGridPreconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridCoarseGridPreconditioner::None:
      string_type = "None";
      break;
    case MultigridCoarseGridPreconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case MultigridCoarseGridPreconditioner::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case MultigridCoarseGridPreconditioner::AMG:
      string_type = "AMG";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerSmoother const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerSmoother::None:
      string_type = "None";
      break;
    case PreconditionerSmoother::PointJacobi:
      string_type = "PointJacobi";
      break;
    case PreconditionerSmoother::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}
