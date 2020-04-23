/*
 * line_plot_calculation.h
 *
 *  Created on: Aug 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_

#include "../../postprocessor/evaluate_solution_in_given_point.h"
#include "../postprocessor/line_plot_data.h"

/*
 *  Evaluate quantities along lines.
 *
 *  Assumptions/Restrictions:
 *
 *   - straight lines, points are distributed equidistantly along the line
 *
 *   - no statistical averaging, instantaneous quantities are calculated
 */
template<int dim, typename Number>
class LinePlotCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  LinePlotCalculator(MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const &                dof_handler_velocity_in,
        DoFHandler<dim> const &                dof_handler_pressure_in,
        Mapping<dim> const &                   mapping_in,
        LinePlotDataInstantaneous<dim> const & line_plot_data_in);

  void
  evaluate(VectorType const & velocity, VectorType const & pressure) const;

private:
  MPI_Comm const & mpi_comm;

  mutable bool clear_files;

  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const>    mapping;

  LinePlotDataInstantaneous<dim> data;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_ */
