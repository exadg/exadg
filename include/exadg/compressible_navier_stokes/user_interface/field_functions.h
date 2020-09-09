/*
 * field_functions.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

template<int dim>
struct FieldFunctions
{
  /*
   * The function initial_solution_velocity is used to initialize the velocity field at the
   * beginning of the simulation.
   */
  std::shared_ptr<Function<dim>> initial_solution;

  std::shared_ptr<Function<dim>> right_hand_side_density;
  std::shared_ptr<Function<dim>> right_hand_side_velocity;
  std::shared_ptr<Function<dim>> right_hand_side_energy;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_ */
