/*
 * lift_and_drag_calculation.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_LIFT_AND_DRAG_CALCULATION_H_
#define INCLUDE_POSTPROCESSOR_LIFT_AND_DRAG_CALCULATION_H_

#include <deal.II/matrix_free/matrix_free.h>

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


template<int dim, typename Number>
class LiftAndDragCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  LiftAndDragCalculator();

  void
  setup(DoFHandler<dim> const &         dof_handler_velocity_in,
        MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const              dof_index_velocity_in,
        unsigned int const              dof_index_pressure_in,
        unsigned int const              quad_index_in,
        LiftAndDragData const &         lift_and_drag_data_in);

  void
  evaluate(VectorType const & velocity, VectorType const & pressure, Number const & time) const;

private:
  mutable bool clear_files_lift_and_drag;

  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  MatrixFree<dim, Number> const *     matrix_free;
  unsigned int                        dof_index_velocity, dof_index_pressure, quad_index;

  mutable double c_L_max, c_D_max;

  LiftAndDragData lift_and_drag_data;
};


#endif /* INCLUDE_POSTPROCESSOR_LIFT_AND_DRAG_CALCULATION_H_ */
