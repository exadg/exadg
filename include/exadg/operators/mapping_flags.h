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

#ifndef INCLUDE_EXADG_OPERATORS_MAPPING_FLAGS_H_
#define INCLUDE_EXADG_OPERATORS_MAPPING_FLAGS_H_

#include <deal.II/fe/fe_update_flags.h>

namespace ExaDG
{
using namespace dealii;

struct MappingFlags
{
  MappingFlags()
    : cells(update_default), inner_faces(update_default), boundary_faces(update_default)
  {
  }

  MappingFlags
  operator||(MappingFlags const & other)
  {
    MappingFlags flags_combined;

    flags_combined.cells          = this->cells | other.cells;
    flags_combined.inner_faces    = this->inner_faces | other.inner_faces;
    flags_combined.boundary_faces = this->boundary_faces | other.boundary_faces;

    return flags_combined;
  }

  UpdateFlags cells;
  UpdateFlags inner_faces;
  UpdateFlags boundary_faces;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_MAPPING_FLAGS_H_ */
