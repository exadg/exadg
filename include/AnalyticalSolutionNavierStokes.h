/*
 * AnalyticalSolutionNavierStokes.h
 *
 *  Created on: Oct 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_ANALYTICALSOLUTIONNAVIERSTOKES_H_
#define INCLUDE_ANALYTICALSOLUTIONNAVIERSTOKES_H_


template<int dim>
struct AnalyticalSolutionNavierStokes
{
  /*
   *  velocity
   */
  std_cxx11::shared_ptr<Function<dim> > velocity;

  /*
   *  pressure
   */
  std_cxx11::shared_ptr<Function<dim> > pressure;
};


#endif /* INCLUDE_ANALYTICALSOLUTIONNAVIERSTOKES_H_ */
