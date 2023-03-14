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
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  DirichletCached,
  Neumann,
  NeumannCached
};

template<int dim>
struct BoundaryDescriptor
{
  // Dirichlet
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // ComponentMask
  // If a certain boundary ID is not inserted into this map, it is assumed that all components are
  // active, in analogy to the default constructor of dealii::ComponentMask.
  std::map<dealii::types::boundary_id, dealii::ComponentMask> dirichlet_bc_component_mask;

  // Another type of Dirichlet boundary condition where the Dirichlet values come
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for the ALE mesh deformation problem in fluid-structure interaction).
  // ComponentMask is not implemented/available for this type of boundary condition.
  std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>> dirichlet_cached_bc;

  // Neumann
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> neumann_bc;

  // another type of Neumann boundary condition where the traction force comes
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Neumann boundary condition
  // is required for fluid-structure interaction problems)
  std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>> neumann_cached_bc;

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->dirichlet_cached_bc.find(boundary_id) != this->dirichlet_cached_bc.end())
      return BoundaryType::DirichletCached;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;
    else if(this->neumann_cached_bc.find(boundary_id) != this->neumann_cached_bc.end())
      return BoundaryType::NeumannCached;

    AssertThrow(false,
                dealii::ExcMessage(
                  "Could not find a boundary type to the specified boundary_id = " +
                  std::to_string(boundary_id) +
                  ". A possible reason is that you "
                  "forgot to define a boundary condition for this boundary_id, or "
                  "that the boundary type associated to this boundary has not been "
                  "implemented."));

    return BoundaryType::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(
      dealii::types::boundary_id const             boundary_id,
      std::set<dealii::types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(dirichlet_bc.find(boundary_id) != dirichlet_bc.end())
    {
      counter++;

      AssertThrow(
        dirichlet_bc_component_mask.find(boundary_id) != dirichlet_bc_component_mask.end(),
        dealii::ExcMessage(
          "dirichlet_bc_component_mask must contain the same boundary IDs as dirichlet_bc."));
    }

    if(dirichlet_cached_bc.find(boundary_id) != dirichlet_cached_bc.end())
      counter++;

    if(neumann_bc.find(boundary_id) != neumann_bc.end())
      counter++;

    if(neumann_cached_bc.find(boundary_id) != neumann_cached_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

} // namespace Structure
} // namespace ExaDG

#endif
