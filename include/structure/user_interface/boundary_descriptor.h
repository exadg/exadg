/*
 * boundary_descriptor.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

using namespace dealii;

namespace Structure
{
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  Neumann
};

template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
  std::map<types::boundary_id, ComponentMask>                  dirichlet_bc_component_mask;

  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryType::Undefined;
  }

  // TODO adapt interfaces of this function to the rest of the code
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<BoundaryType, std::shared_ptr<Function<dim>>>
    get_boundary(types::boundary_id const & boundary_id) const
  {
    // process dirichlet bc
    {
      auto res = this->dirichlet_bc.find(boundary_id);

      if(res != this->dirichlet_bc.end())
        return {BoundaryType::Dirichlet, res->second};
    }

    // process neumann bc
    {
      auto res = this->neumann_bc.find(boundary_id);
      if(res != this->neumann_bc.end())
        return {BoundaryType::Neumann, res->second};
    }

    std::cout << boundary_id << std::endl;
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return {BoundaryType::Undefined,
            std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim))};
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

    if(neumann_bc.find(boundary_id) != neumann_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

} // namespace Structure

#endif
