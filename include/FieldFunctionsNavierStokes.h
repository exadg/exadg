/*
 * FieldFunctionsNavierStokes.h
 *
 *  Created on: Aug 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_FIELDFUNCTIONSNAVIERSTOKES_H_
#define INCLUDE_FIELDFUNCTIONSNAVIERSTOKES_H_


template<int dim>
struct FieldFunctionsNavierStokes
{
  std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
  std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
};


#endif /* INCLUDE_FIELDFUNCTIONSNAVIERSTOKES_H_ */
