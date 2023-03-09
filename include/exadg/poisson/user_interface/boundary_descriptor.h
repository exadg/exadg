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

#ifndef INCLUDE_EXADG_POISSON_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_EXADG_POISSON_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/function_cached.h>

namespace ExaDG
{
namespace Poisson
{
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  DirichletCached,
  Neumann
};

template<int rank, int dim>
struct BoundaryDescriptor
{
  // Dirichlet
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // ComponentMaskn (only used/relevant for continuous Galerkin, ignored for DG)
  // If a certain boundary ID is not inserted into this map, it is assumed that all components are
  // active, in analogy to the default constructor of dealii::ComponentMask.
  std::map<dealii::types::boundary_id, dealii::ComponentMask> dirichlet_bc_component_mask;

  // Another type of Dirichlet boundary condition where the Dirichlet values come
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for the ALE mesh deformation problem in fluid-structure interaction).
  // ComponentMask is not implemented/available for this type of boundary condition.
  std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<rank, dim>>>
    dirichlet_cached_bc;

  // Neumann
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> neumann_bc;

  // returns the boundary type
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

    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

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
      counter++;

    if(dirichlet_cached_bc.find(boundary_id) != dirichlet_cached_bc.end())
      counter++;

    if(neumann_bc.find(boundary_id) != neumann_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
