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

#ifndef INCLUDE_FUNCTIONALITIES_MAPPING_DEGREE_H_
#define INCLUDE_FUNCTIONALITIES_MAPPING_DEGREE_H_

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/grid/enum_types.h>

namespace ExaDG
{
using namespace dealii;

inline unsigned int
get_mapping_degree(MappingType const & mapping_type, unsigned int const degree_shape_functions)
{
  unsigned int degree = 0;

  switch(mapping_type)
  {
    case MappingType::Affine:
      degree = 1;
      break;
    case MappingType::Quadratic:
      degree = 2;
      break;
    case MappingType::Cubic:
      degree = 3;
      break;
    case MappingType::Isoparametric:
      degree = degree_shape_functions;
      break;
    default:
      AssertThrow(false, ExcMessage("Not implemented."));
      break;
  }

  return degree;
}

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MAPPING_DEGREE_H_ */
