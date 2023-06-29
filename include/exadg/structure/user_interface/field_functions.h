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


#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_FIELD_FUNCTIONS_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_FIELD_FUNCTIONS_H_

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct FieldFunctions
{
  std::shared_ptr<dealii::Function<dim>> right_hand_side;
  std::shared_ptr<dealii::Function<dim>> initial_displacement;
  std::shared_ptr<dealii::Function<dim>> initial_velocity;

  // Initial acceleration: Either prescribe the initial acceleration as a volume field or - if this
  // is unknown - prescribe the initial acceleration on the Dirichlet boundary, while the initial
  // acceleration in the volume will be derived as a numerical solution of the PDE problem.

  // Initial acceleration as volume field:
  std::shared_ptr<dealii::Function<dim>> initial_acceleration;

  // Initial acceleration prescribed on Dirichlet boundary:
  // This data structure will only be used if the above pointer to initial_acceleration is
  // uninitialized.
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    initial_acceleration_dirichlet_boundary;
};

} // namespace Structure
} // namespace ExaDG

#endif
