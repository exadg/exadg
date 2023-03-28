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
#include <exadg/structure/user_interface/enum_types.h>

namespace ExaDG
{
namespace Structure
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/


std::string
enum_to_string(ProblemType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case ProblemType::Undefined:
      string_type = "Undefined";
      break;
    case ProblemType::Steady:
      string_type = "Steady";
      break;
    case ProblemType::QuasiStatic:
      string_type = "QuasiStatic";
      break;
    case ProblemType::Unsteady:
      string_type = "Unsteady";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(Type2D const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case Type2D::Undefined:
      string_type = "Undefined";
      break;
    case Type2D::PlaneStress:
      string_type = "PlaneStress";
      break;
    case Type2D::PlaneStrain:
      string_type = "PlaneStrain";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MaterialType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MaterialType::Undefined:
      string_type = "Undefined";
      break;
    case MaterialType::StVenantKirchhoff:
      string_type = "StVenantKirchhoff";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                                 PHYSICAL QUANTITIES                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                             TEMPORAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



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
    case Preconditioner::None:
      string_type = "None";
      break;
    case Preconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case Preconditioner::Multigrid:
      string_type = "Multigrid";
      break;
    case Preconditioner::AMG:
      string_type = "AMG";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(LinearizationType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case LinearizationType::Undefined:
      string_type = "Undefined";
      break;
    case LinearizationType::Newton:
      string_type = "Newton";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

} // namespace Structure
} // namespace ExaDG
