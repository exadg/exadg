/*
 * lift_and_drag_data.h
 *
 *  Created on: Oct 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LIFT_AND_DRAG_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LIFT_AND_DRAG_DATA_H_

#include <set>

#include <deal.II/base/types.h>

using namespace dealii;

struct LiftAndDragData
{
  LiftAndDragData()
    : calculate_lift_and_drag(false),
      viscosity(1.0),
      reference_value(1.0),
      filename_lift("lift"),
      filename_drag("drag")
  {
  }

  /*
   *  active or not
   */
  bool calculate_lift_and_drag;

  /*
   *  Kinematic viscosity
   */
  double viscosity;

  /*
   *  c_L = L / (1/2 rho U^2 A) = L / (reference_value)
   */
  double reference_value;

  /*
   *  set containing boundary ID's of the surface area used
   *  to calculate lift and drag coefficients
   */
  std::set<types::boundary_id> boundary_IDs;

  /*
   *  filenames
   */
  std::string filename_lift;
  std::string filename_drag;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LIFT_AND_DRAG_DATA_H_ */
