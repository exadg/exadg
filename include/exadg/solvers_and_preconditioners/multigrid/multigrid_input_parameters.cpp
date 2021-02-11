/*
 * multigrid_input_parameters.cpp
 *
 *  Created on: Feb 1, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_input_parameters.h>

namespace ExaDG
{
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
    case MultigridType::chMG:
      string_type = "ch-MG";
      break;
    case MultigridType::hcMG:
      string_type = "hc-MG";
      break;
    case MultigridType::pMG:
      string_type = "p-MG";
      break;
    case MultigridType::cpMG:
      string_type = "cp-MG";
      break;
    case MultigridType::pcMG:
      string_type = "pc-MG";
      break;
    case MultigridType::hpMG:
      string_type = "hp-MG";
      break;
    case MultigridType::chpMG:
      string_type = "chp-MG";
      break;
    case MultigridType::hcpMG:
      string_type = "hcp-MG";
      break;
    case MultigridType::hpcMG:
      string_type = "hpc-MG";
      break;
    case MultigridType::phMG:
      string_type = "ph-MG";
      break;
    case MultigridType::cphMG:
      string_type = "cph-MG";
      break;
    case MultigridType::pchMG:
      string_type = "pch-MG";
      break;
    case MultigridType::phcMG:
      string_type = "phc-MG";
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

bool
MultigridData::involves_h_transfer() const
{
  if(type == MultigridType::hMG || type == MultigridType::chMG || type == MultigridType::phMG ||
     type == MultigridType::hcMG || type == MultigridType::hpMG || type == MultigridType::chpMG ||
     type == MultigridType::cphMG || type == MultigridType::hcpMG || type == MultigridType::hpcMG ||
     type == MultigridType::pchMG || type == MultigridType::phcMG)
    return true;
  else
    return false;
}
} // namespace ExaDG
