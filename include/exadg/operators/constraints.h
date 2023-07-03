/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_OPERATORS_CONSTRAINTS_H_
#define INCLUDE_EXADG_OPERATORS_CONSTRAINTS_H_

// deal.II
#include <deal.II/dofs/dof_tools.h>

// ExaDG
#include <exadg/grid/grid_utilities.h>

namespace ExaDG
{
/**
 * This function adds hanging-node and periodicity constraints. This function combines these two
 * types of constraints since deal.II requires to add these constraints in a certain order.
 */
template<int dim, typename Number>
void
add_hanging_node_and_periodicity_constraints(dealii::AffineConstraints<Number> & affine_constraints,
                                             Grid<dim> const &                   grid,
                                             dealii::DoFHandler<dim> const &     dof_handler)
{
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  affine_constraints.reinit(locally_relevant_dofs);

  // hanging nodes (needs to be done before imposing periodicity constraints
  if(grid.triangulation->has_hanging_nodes())
  {
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, affine_constraints);
  }

  // constraints from periodic boundary conditions
  if(not(grid.periodic_face_pairs.empty()))
  {
    auto periodic_faces_dof =
      GridUtilities::transform_periodic_face_pairs_to_dof_cell_iterator(grid.periodic_face_pairs,
                                                                        dof_handler);

    dealii::DoFTools::make_periodicity_constraints<dim, dim, Number>(periodic_faces_dof,
                                                                     affine_constraints);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_CONSTRAINTS_H_ */
