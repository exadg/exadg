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

#ifndef INCLUDE_FUNCTIONALITIES_VERIFY_BOUNDARY_CONDITIONS_H_
#define INCLUDE_FUNCTIONALITIES_VERIFY_BOUNDARY_CONDITIONS_H_

// ExaDG
#include <exadg/grid/grid.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename BoundaryDescriptor>
void
verify_boundary_conditions(BoundaryDescriptor const & boundary_descriptor,
                           Grid<dim, Number> const &  grid)
{
  // fill set with periodic boundary ids
  std::set<types::boundary_id> periodic_boundary_ids;
  for(auto periodic_pair : grid.periodic_faces)
  {
    AssertThrow(periodic_pair.cell[0]->level() == 0,
                ExcMessage("Received periodic face pair on non-zero level"));

    periodic_boundary_ids.insert(
      periodic_pair.cell[0]->face(periodic_pair.face_idx[0])->boundary_id());
    periodic_boundary_ids.insert(
      periodic_pair.cell[1]->face(periodic_pair.face_idx[1])->boundary_id());
  }

  // Make sure that each boundary face has exactly one boundary type
  for(auto cell : *grid.triangulation)
  {
    for(const unsigned int f : cell.face_indices())
    {
      if(cell.at_boundary(f))
      {
        types::boundary_id const boundary_id = cell.face(f)->boundary_id();
        boundary_descriptor.verify_boundary_conditions(boundary_id, periodic_boundary_ids);
      }
    }
  }
}

} // namespace ExaDG


#endif /* INCLUDE_FUNCTIONALITIES_VERIFY_BOUNDARY_CONDITIONS_H_ */
