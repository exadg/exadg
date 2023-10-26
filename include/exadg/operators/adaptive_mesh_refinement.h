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
 *
 * Source: https://github.com/MeltPoolDG/MeltPoolDG
 * Author: Peter Munch, Magdalena Schreter, TUM, December 2020
 */

#ifndef INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_
#define INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/solution_transfer.h>

namespace ExaDG
{
struct AdaptiveMeshRefinementData
{
  bool         do_not_modify_boundary_cells = false;
  double       upper_perc_to_refine         = 0.0;
  double       lower_perc_to_coarsen        = 0.0;
  int          every_n_step                 = 1;
  unsigned int refine_space_max             = 10;
  int          refine_space_min             = 0;
};

inline bool
trigger_coarsening_and_refinement_now(const AdaptiveMeshRefinementData & amr_data,
                                      const int                          time_step_number)
{
  return ((time_step_number == 0) or !(time_step_number % amr_data.every_n_step));
}

/*
 * Limit the coarsening and refinement by adapting the flags set in the triangulation.
 * This considers the maximum and minimum refinement levels provided and might remove
 * refinement flags set on the boundary if requested.
 */
template<int dim>
void
limit_coarsening_and_refinement(dealii::Triangulation<dim> &       tria,
                                AdaptiveMeshRefinementData const & amr_data)
{
  //  Limit the maximum refinement levels of cells of the grid.
  if(tria.n_levels() > amr_data.refine_space_max)
  {
    for(auto & cell : tria.active_cell_iterators_on_level(amr_data.refine_space_max))
    {
      cell->clear_refine_flag();
    }
  }

  //  Limit the minimum refinement levels of cells of the grid.
  for(auto & cell : tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      if(cell->level() <= amr_data.refine_space_min)
      {
        cell->clear_coarsen_flag();
      }

      // Do not coarsen/refine cells along boundary if requested
      if(amr_data.do_not_modify_boundary_cells)
      {
        for(auto & face : cell->face_iterators())
        {
          if(face->at_boundary())
          {
            if(cell->refine_flag_set())
            {
              cell->clear_refine_flag();
            }
            else
            {
              cell->clear_coarsen_flag();
            }
          }
        }
      }
    }
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_ */
