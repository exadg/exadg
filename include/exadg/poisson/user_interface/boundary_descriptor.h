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
  DirichletMortar,
  Neumann
};

template<int rank, int dim>
struct BoundaryDescriptor
{
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // dealii::ComponentMask is only used for continuous elements, and is ignored for DG
  std::map<dealii::types::boundary_id, dealii::ComponentMask> dirichlet_bc_component_mask;

  std::map<dealii::types::boundary_id, std::shared_ptr<FunctionCached<rank, dim>>>
    dirichlet_mortar_bc;

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> neumann_bc;

  // returns the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->dirichlet_mortar_bc.find(boundary_id) != this->dirichlet_mortar_bc.end())
      return BoundaryType::DirichletMortar;
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

    if(dirichlet_mortar_bc.find(boundary_id) != dirichlet_mortar_bc.end())
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
