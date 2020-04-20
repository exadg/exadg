/*
 * boundary_descriptor.h
 *
 *  Created on: 19.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_POISSON_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include "../../functionalities/function_interpolation.h"

using namespace dealii;

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
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;

  // ComponentMask is only used for continuous elements, and is ignored for DG
  std::map<types::boundary_id, ComponentMask> dirichlet_bc_component_mask;

  std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<rank, dim>>>
    dirichlet_mortar_bc;

  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  // returns the boundary type
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

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryType::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(types::boundary_id const             boundary_id,
                               std::set<types::boundary_id> const & periodic_boundary_ids) const
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

    AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

} // namespace Poisson



#endif /* INCLUDE_POISSON_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
