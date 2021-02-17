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

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/component_mask.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/function_cached.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

enum class BoundaryType
{
  Undefined,
  Dirichlet,
  DirichletMortar,
  Neumann,
  NeumannMortar
};

template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
  std::map<types::boundary_id, ComponentMask>                  dirichlet_bc_component_mask;

  // another type of Dirichlet boundary condition where the Dirichlet values come
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for fluid-structure interaction problems)
  std::map<types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>> dirichlet_mortar_bc;

  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  // another type of Neumann boundary condition where the traction force comes
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Neumann boundary condition
  // is required for fluid-structure interaction problems)
  std::map<types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>> neumann_mortar_bc;

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->dirichlet_mortar_bc.find(boundary_id) != this->dirichlet_mortar_bc.end())
      return BoundaryType::DirichletMortar;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;
    else if(this->neumann_mortar_bc.find(boundary_id) != this->neumann_mortar_bc.end())
      return BoundaryType::NeumannMortar;

    AssertThrow(false,
                ExcMessage("Could not find a boundary type to the specified boundary_id = " +
                           std::to_string(boundary_id) +
                           ". A possible reason is that you "
                           "forgot to define a boundary condition for this boundary_id, or "
                           "that the boundary type associated to this boundary has not been "
                           "implemented."));

    return BoundaryType::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(types::boundary_id const             boundary_id,
                               std::set<types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(dirichlet_bc.find(boundary_id) != dirichlet_bc.end())
    {
      counter++;

      AssertThrow(
        dirichlet_bc_component_mask.find(boundary_id) != dirichlet_bc_component_mask.end(),
        ExcMessage(
          "dirichlet_bc_component_mask must contain the same boundary IDs as dirichlet_bc."));
    }

    if(dirichlet_mortar_bc.find(boundary_id) != dirichlet_mortar_bc.end())
      counter++;

    if(neumann_bc.find(boundary_id) != neumann_bc.end())
      counter++;

    if(neumann_mortar_bc.find(boundary_id) != neumann_mortar_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

} // namespace Structure
} // namespace ExaDG

#endif
