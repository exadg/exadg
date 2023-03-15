/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/poisson/user_interface/enum_types.h>

namespace ExaDG
{
namespace Poisson
{
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
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
    case Solver::GMRES:
      string_type = "GMRES";
      break;
    case Solver::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
    case Preconditioner::AMG:
      string_type = "AMG";
      break;
    case Preconditioner::Multigrid:
      string_type = "Multigrid";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


} // namespace Poisson
} // namespace ExaDG
