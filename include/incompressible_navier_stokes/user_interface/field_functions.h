/*
 * field_functions.h
 *
 *  Created on: Aug 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_

#include "../../../applications/grid_tools/mesh_movement_functions.h"

namespace IncNS
{
template<int dim>
struct FieldFunctions
{
  /*
   * The function initial_solution_velocity is used to initialize the velocity field at the
   * beginning of the simulation.
   */
  std::shared_ptr<Function<dim>> initial_solution_velocity;

  /*
   * The function initial_solution_pressure is used to initialize the pressure field at the
   * beginning of the simulation.
   */
  std::shared_ptr<Function<dim>> initial_solution_pressure;

  /*
   * The function analytical_solution_pressure is used to adjust the pressure level in the special
   * case of ...
   *   ... pure Dirichlet boundary conditions (where the pressure is only defined up to an additive
   * constant) and
   *   ... if an analytical solution for the pressure is available.
   */
  std::shared_ptr<Function<dim>> analytical_solution_pressure;

  /*
   * The function right_hand_side is used to evaluate the body force term on the right-hand side of
   * the momentum equation of the incompressible Navier-Stokes equations.
   */
  std::shared_ptr<Function<dim>> right_hand_side;

  /*
   * A function that describes a mesh movement analytically and that is used in case of an
   * Arbitrary Lagrangian-Eulerian formulation.
   */
  std::shared_ptr<MeshMovementFunctions<dim>> mesh_movement;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_ */
