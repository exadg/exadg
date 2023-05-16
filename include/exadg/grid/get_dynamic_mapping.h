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

#ifndef INCLUDE_EXADG_GRID_GET_DYNAMIC_MAPPING_H_
#define INCLUDE_EXADG_GRID_GET_DYNAMIC_MAPPING_H_

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_motion_base.h>

namespace ExaDG
{
/**
 * Return pointer to dynamic mapping (and redirect to static mapping if dynamic mapping is not
 * initialized).
 */
template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
get_dynamic_mapping(std::shared_ptr<Grid<dim> const>                   grid,
                    std::shared_ptr<GridMotionBase<dim, Number> const> grid_motion)
{
  if(grid_motion.get() != 0)
  {
    return grid_motion->get_mapping();
  }
  else
  {
    AssertThrow(grid->mapping.get() != 0,
                dealii::ExcMessage("Uninitialized mapping object detected."));

    return grid->mapping;
  }
}
} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_GET_DYNAMIC_MAPPING_H_ */
