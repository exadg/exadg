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

/**
 * This utility function extracts the keys of a map provided as second argument and inserts these
 * keys into the set provided as first argument.
 */
template<typename Key, typename Data>
inline void
fill_keys_of_map_into_set(std::set<Key> & set, std::map<Key, Data> const & map)
{
  for(auto iter : map)
  {
    set.insert(iter.first);
  }
}

/**
 * This function inserts additional boundary IDs together with a default dealii::ComponentMask into
 * map_bid_to_mask in case that the given boundary ID is not already an element of map_bid_to_mask.
 */
inline void
fill_map_bid_to_mask_with_default_mask(
  std::map<dealii::types::boundary_id, dealii::ComponentMask> & map_bid_to_mask,
  std::set<dealii::types::boundary_id> const &                  set_boundary_ids)
{
  for(auto const & it : set_boundary_ids)
  {
    // use default mask if no maks has been defined
    if(map_bid_to_mask.find(it) == map_bid_to_mask.end())
      map_bid_to_mask.insert({it, dealii::ComponentMask()});
  }
}

/**
 * This function calls dealii::DoFTools::make_zero_boundary_constraints() for all boundary IDs with
 * corresponding dealii::ComponentMask handed over to this function. This is necessary as a utility
 * function in ExaDG, since deal.II does not allow to provide independent component masks for each
 * boundary ID in a set of boundary IDs.
 */
template<int dim, typename Number>
void
add_homogeneous_dirichlet_constraints(
  dealii::AffineConstraints<Number> & affine_constraints,
  dealii::DoFHandler<dim> const &     dof_handler,
  std::map<dealii::types::boundary_id, dealii::ComponentMask> const &
    map_boundary_id_to_component_mask)
{
  for(auto const & it : map_boundary_id_to_component_mask)
  {
    dealii::DoFTools::make_zero_boundary_constraints(dof_handler,
                                                     it.first,
                                                     affine_constraints,
                                                     it.second);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_CONSTRAINTS_H_ */
