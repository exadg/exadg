/*
 * FieldFunctionsCompNavierStokes.h
 *
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_

namespace CompNS
{

template<int dim>
struct FieldFunctions
{
/*
   *  The function initial_solution_velocity is used to initialize
   *  the velocity field at the beginning of the simulation.
   */
  std::shared_ptr<Function<dim> > initial_solution;

  std::shared_ptr<Function<dim> > right_hand_side_density;
  std::shared_ptr<Function<dim> > right_hand_side_velocity;
  std::shared_ptr<Function<dim> > right_hand_side_energy;
};

}


#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_FIELD_FUNCTIONS_H_ */
