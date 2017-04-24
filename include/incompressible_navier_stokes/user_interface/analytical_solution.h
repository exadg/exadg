/*
 * AnalyticalSolutionNavierStokes.h
 *
 *  Created on: Oct 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ANALYTICAL_SOLUTION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ANALYTICAL_SOLUTION_H_

#include <deal.II/base/function.h>

using namespace dealii;

template<int dim>
struct AnalyticalSolutionNavierStokes
{
  /*
   *  velocity
   */
  std::shared_ptr<Function<dim> > velocity;

  /*
   *  pressure
   */
  std::shared_ptr<Function<dim> > pressure;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ANALYTICAL_SOLUTION_H_ */
