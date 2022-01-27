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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct FieldFunctions
{
  /*
   * The function initial_solution_velocity is used to initialize the velocity field at the
   * beginning of the simulation.
   */
  std::shared_ptr<dealii::Function<dim>> initial_solution_velocity;

  /*
   * The function initial_solution_pressure is used to initialize the pressure field at the
   * beginning of the simulation.
   */
  std::shared_ptr<dealii::Function<dim>> initial_solution_pressure;

  /*
   * The function analytical_solution_pressure is used to adjust the pressure level in the special
   * case of ...
   *   ... pure Dirichlet boundary conditions (where the pressure is only defined up to an additive
   * constant) and
   *   ... if an analytical solution for the pressure is available.
   */
  std::shared_ptr<dealii::Function<dim>> analytical_solution_pressure;

  /*
   * The function right_hand_side is used to evaluate the body force term on the right-hand side of
   * the momentum equation of the incompressible Navier-Stokes equations.
   */
  std::shared_ptr<dealii::Function<dim>> right_hand_side;
  std::shared_ptr<dealii::Function<dim>> gravitational_force; // Boussinesq term
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_ */
