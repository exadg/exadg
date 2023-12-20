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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

#include <set>

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>

namespace ExaDG
{
namespace Acoustics
{
/* Boundary conditions:
 *
 * +---------------------------+-------------------------+-------------------------+
 * | example                   | pressure                | velocity                |
 * +---------------------------+-------------------------+-------------------------+
 * | prescribe pressure values | Dirichlet:              |                         |
 * |                           | prescribe g_p           | no BCs to be prescribed |
 * +-----------------------------------------------------+-------------------------+
 * | prescribe velocity values |                         | Dirichlet:              |
 * |                           | no BCs to be prescribed | prescribe g_u           |
 * +---------------------------+-------------------------+-------------------------+
 * | admittance BC             |                         | Admittance:             |
 * |                           | no BCs to be prescribed | prescribe Y             |
 * +---------------------------+-------------------------+-------------------------+
 */

enum class BoundaryType
{
  Undefined,
  PressureDirichlet,
  VelocityDirichlet,
  Admittance
};

template<int dim>
struct BoundaryDescriptor
{
  using boundary_type = BoundaryType;

  static constexpr int dimension = dim;

  // Dirichlet: prescribe pressure
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> pressure_dbc;

  // Dirichlet: prescribe velocity
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> velocity_dbc;

  // BC for Admittance Y:
  // Special cases are:
  // Y = 0: sound hard (perfectly reflecting)
  // Y = 1: first order ABC
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> admittance_bc;

  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->pressure_dbc.find(boundary_id) != this->pressure_dbc.end())
      return BoundaryType::PressureDirichlet;

    if(this->velocity_dbc.find(boundary_id) != this->velocity_dbc.end())
      return BoundaryType::VelocityDirichlet;

    if(this->admittance_bc.find(boundary_id) != this->admittance_bc.end())
      return BoundaryType::Admittance;

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
    if(this->pressure_dbc.find(boundary_id) != this->pressure_dbc.end())
      counter++;

    if(this->velocity_dbc.find(boundary_id) != this->velocity_dbc.end())
      counter++;

    if(this->admittance_bc.find(boundary_id) != this->admittance_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
