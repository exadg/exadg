/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DOF_TOOLS_EXTENSION_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DOF_TOOLS_EXTENSION_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

DEAL_II_NAMESPACE_OPEN

namespace DoFTools
{
/**
 * Similar to map_dofs_to_support_points, but restricted to the boundary of
 * the given boundary ID
 */
template<int dim, int spacedim>
void
map_boundary_dofs_to_support_points(
  Mapping<dim, spacedim> const &                       mapping,
  DoFHandler<dim, spacedim> const &                    dof_handler,
  std::map<types::global_dof_index, Point<spacedim>> & support_points,
  ComponentMask const &                                in_mask,
  types::boundary_id const                             boundary_id)
{
  FiniteElement<dim, spacedim> const & fe = dof_handler.get_fe();
  // check whether every fe in the collection has support points
  Assert(fe.has_support_points(), typename FiniteElement<dim>::ExcFEHasNoSupportPoints());

  Quadrature<dim - 1> const quad(fe.get_unit_face_support_points());

  // Take care of components
  ComponentMask const mask =
    (in_mask.size() == 0 ? ComponentMask(fe.n_components(), true) : in_mask);

  // Now loop over all cells and enquire the support points on each
  // of these. we use dummy quadrature formulas where the quadrature
  // points are located at the unit support points to enquire the
  // location of the support points in real space.
  //
  // The weights of the quadrature rule have been set to invalid
  // values by the used constructor.
  FEFaceValues<dim, spacedim> fe_values(mapping, fe, quad, update_quadrature_points);

  std::vector<types::global_dof_index> local_dof_indices;
  for(auto const & cell : dof_handler.active_cell_iterators())
    if(cell->is_locally_owned())
      for(auto const & face : cell->face_iterators())
        if(face->at_boundary() == true && face->boundary_id() == boundary_id)
        // only work on locally relevant cells
        {
          fe_values.reinit(cell, face);

          local_dof_indices.resize(fe.dofs_per_face);
          face->get_dof_indices(local_dof_indices);

          std::vector<Point<spacedim>> const & points = fe_values.get_quadrature_points();

          for(unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
          {
            unsigned int const dof_comp = fe.face_system_to_component_index(i).first;

            // insert the values into the map if it is a valid component
            if(mask[dof_comp])
              support_points[local_dof_indices[i]] = points[i];
          }
        }
}
} // namespace DoFTools

DEAL_II_NAMESPACE_CLOSE

#endif
