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

#ifndef INCLUDE_EXADG_GRID_PERFORM_LOCAL_REFINEMENTS_H_
#define INCLUDE_EXADG_GRID_PERFORM_LOCAL_REFINEMENTS_H_

// C/C++
#include <vector>

// deal.II
#include <deal.II/grid/tria.h>

namespace ExaDG
{
/**
 * This function performs local refinements on a given triangulation, where the number of
 * local refinements is described by a vector containing the number of local refinements for
 * each material id, i.e. vector[material_id] = n_local_refinements.
 *
 * TODO: change the design of this function from a vector towards a map from material IDs to
 * the number of local refinements.
 */
template<int dim>
void
refine_local(dealii::Triangulation<dim> &      tria,
             std::vector<unsigned int> const & vector_local_refinements)
{
  std::vector<unsigned int> refine_local = vector_local_refinements;

  // execute refinement until every refinement counter has been decreased to 0
  while(*max_element(refine_local.begin(), refine_local.end()) > 0)
  {
    // loop over all material IDs in refinement vector
    for(size_t material_id = 0; material_id < refine_local.size(); material_id++)
    {
      // only if cells with current material_id have to be refined
      if(refine_local[material_id] > 0)
      {
        for(auto & cell : tria.active_cell_iterators())
        {
          if(cell->material_id() == material_id)
          {
            cell->set_refine_flag();
          }
        }

        // decrease refinement counter
        refine_local[material_id]--;
      }
    }

    // execute local refinement
    tria.execute_coarsening_and_refinement();
  }
}

} // namespace ExaDG


#endif /* INCLUDE_EXADG_GRID_PERFORM_LOCAL_REFINEMENTS_H_ */
