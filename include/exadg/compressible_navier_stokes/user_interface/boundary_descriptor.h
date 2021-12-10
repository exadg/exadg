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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

using namespace dealii;

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

// ExaDG
#include <exadg/compressible_navier_stokes/user_interface/parameters.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;


enum class BoundaryType
{
  Undefined,
  Dirichlet,
  Neumann
};

template<int dim>
struct BoundaryDescriptorStd
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

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(types::boundary_id const             boundary_id,
                               std::set<types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      counter++;

    if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

template<int dim>
struct BoundaryDescriptorEnergy : public BoundaryDescriptorStd<dim>
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

template<int dim>
struct BoundaryDescriptor
{
  BoundaryDescriptorStd<dim>    density;
  BoundaryDescriptorStd<dim>    velocity;
  BoundaryDescriptorStd<dim>    pressure;
  BoundaryDescriptorEnergy<dim> energy;
};

template<int dim, typename Number>
inline void
verify_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
                           Grid<dim, Number> const &                      grid)
{
  ExaDG::verify_boundary_conditions(boundary_descriptor->density, grid);
  ExaDG::verify_boundary_conditions(boundary_descriptor->velocity, grid);
  ExaDG::verify_boundary_conditions(boundary_descriptor->pressure, grid);
  ExaDG::verify_boundary_conditions(boundary_descriptor->energy, grid);
}

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
