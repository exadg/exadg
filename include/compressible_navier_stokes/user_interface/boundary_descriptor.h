/*
 * boundary_descriptor.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include "input_parameters.h"

namespace CompNS
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
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  // return the boundary type
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
};

template<int dim>
struct BoundaryDescriptorEnergy : public BoundaryDescriptor<dim>
{
  std::map<types::boundary_id, EnergyBoundaryVariable> boundary_variable;

  // return the boundary variable
  inline DEAL_II_ALWAYS_INLINE //
    EnergyBoundaryVariable
    get_boundary_variable(types::boundary_id const & boundary_id) const
  {
    EnergyBoundaryVariable boundary_variable = this->boundary_variable.find(boundary_id)->second;

    AssertThrow(boundary_variable != EnergyBoundaryVariable::Undefined,
                ExcMessage("Energy boundary variable is undefined!"));

    return boundary_variable;
  }
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
