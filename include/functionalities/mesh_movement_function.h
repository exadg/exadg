/*
 * mesh_movement_function.h
 *
 *  Created on: Nov 2, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_MESH_MOVEMENT_FUNCTION_H_
#define INCLUDE_FUNCTIONALITIES_MESH_MOVEMENT_FUNCTION_H_

using namespace dealii;

template<int dim>
class MeshMovementFunction : public Function<dim>
{
public:
  MeshMovementFunction() : Function<dim>(dim, 0.0)
  {
  }

  virtual ~MeshMovementFunction()
  {
  }

  virtual double
  displacement(Point<dim> const & x, unsigned int const coordinate_direction = 0) const = 0;

  /*
   * The function value() returns the velocity. The name value() originates from the fact that
   * VectorTools::interpolate() expects the Function<dim> class to have a member function with this
   * name.
   */
  virtual double
  value(Point<dim> const & p, unsigned int const component = 0) const = 0;
};



#endif /* INCLUDE_FUNCTIONALITIES_MESH_MOVEMENT_FUNCTION_H_ */
