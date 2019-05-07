/*
 * pressure_difference_data.h
 *
 *  Created on: Oct 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PRESSURE_DIFFERENCE_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PRESSURE_DIFFERENCE_DATA_H_

#include <deal.II/base/point.h>

template<int dim>
struct PressureDifferenceData
{
  PressureDifferenceData() : calculate_pressure_difference(false), filename("pressure_difference")
  {
  }

  /*
   *  active or not
   */
  bool calculate_pressure_difference;

  /*
   *  Points:
   *  calculation of pressure difference: p(point_1) - p(point_2)
   */
  Point<dim> point_1;
  Point<dim> point_2;

  /*
   *  filenames
   */
  std::string filename;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PRESSURE_DIFFERENCE_DATA_H_ */
