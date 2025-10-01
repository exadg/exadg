/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_POSTPROCESSOR_LIFT_AND_DRAG_CALCULATION_H_
#define EXADG_POSTPROCESSOR_LIFT_AND_DRAG_CALCULATION_H_

// deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
struct LiftAndDragData
{
  LiftAndDragData()
    : viscosity(1.0),
      reference_value(1.0),
      directory("output/"),
      filename_lift("lift"),
      filename_drag("drag")
  {
  }

  /*
   *  Data to control output: Set is_active also in the unsteady case
   */
  TimeControlData time_control_data;

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
  std::set<dealii::types::boundary_id> boundary_IDs;

  /*
   *  filenames
   */
  std::string directory;
  std::string filename_lift;
  std::string filename_drag;

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady) const;
};

template<int dim, typename Number>
class LiftAndDragCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  LiftAndDragCalculator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const &         dof_handler_velocity_in,
        dealii::MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const                      dof_index_velocity_in,
        unsigned int const                      dof_index_pressure_in,
        unsigned int const                      quad_index_in,
        LiftAndDragData const &                 lift_and_drag_data_in);

  void
  evaluate(VectorType const & velocity, VectorType const & pressure, double const time) const;

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  mutable bool clear_files;

  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler_velocity;
  dealii::MatrixFree<dim, Number> const *                matrix_free;
  unsigned int dof_index_velocity, dof_index_pressure, quad_index;

  mutable double c_L_min, c_L_max, c_D_min, c_D_max;

  LiftAndDragData data;
};

} // namespace ExaDG

#endif /* EXADG_POSTPROCESSOR_LIFT_AND_DRAG_CALCULATION_H_ */
