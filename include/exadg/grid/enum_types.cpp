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
#include <exadg/grid/enum_types.h>

namespace ExaDG
{
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

} // namespace ExaDG
