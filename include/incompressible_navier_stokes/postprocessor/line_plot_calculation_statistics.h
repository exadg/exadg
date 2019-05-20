/*
 * line_plot_calculation_statistics.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>


#include "line_plot_calculation.h"
#include "line_plot_data.h"

using namespace dealii;

/*
 * This function calculates statistics along lines by averaging over time.
 *
 * Additionally, averaging in circumferential direction can be performed if desired.
 *
 * General assumptions:
 *
 *  - we assume straight lines and an equidistant distribution of the evaluation points along each
 *    line.
 *
 * Assumptions for averaging in circumferential direction:
 *
 *  - to define the plane in which we want to perform the averaging in circumferential direction,
 *    a normal vector has to be specified that has to be oriented normal to the straight line and
 *    normal to the averaging plane. To construct the sample points for averaging in circumferential
 *    direction, we assume that the first point of the line (line.begin) defines the center of the
 *    circle, and that the other points in circumferential direction can be constructed by rotating
 *    the vector from the center of the circle (line.begin) to the current point along the line
 *    around the normal vector.
 */

template<int dim>
class LinePlotCalculatorStatistics
{
public:
  typedef LinearAlgebra::distributed::Vector<double> VectorType;

  typedef
    typename std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>>
      TYPE;

  LinePlotCalculatorStatistics(const DoFHandler<dim> & dof_handler_velocity_in,
                               const DoFHandler<dim> & dof_handler_pressure_in,
                               const Mapping<dim> &    mapping_in);

  void
  setup(LinePlotData<dim> const & line_plot_data_in);

  void
  evaluate(VectorType const &   velocity,
           VectorType const &   pressure,
           double const &       time,
           unsigned int const & time_step_number);

private:
  void
  print_headline(std::ofstream & f, const unsigned int number_of_samples) const
  {
    f << "number of samples: N = " << number_of_samples << std::endl;
  }

  void
  do_evaluate(VectorType const & velocity, VectorType const & pressure);

  void
  do_evaluate_velocity(VectorType const & velocity,
                       Line<dim> const &  line,
                       unsigned int const line_iterator);

  void
  do_evaluate_pressure(VectorType const & pressure,
                       Line<dim> const &  line,
                       unsigned int const line_iterator);

  void
  do_write_output() const;

  mutable bool clear_files;

  DoFHandler<dim> const & dof_handler_velocity;
  DoFHandler<dim> const & dof_handler_pressure;
  Mapping<dim> const &    mapping;
  MPI_Comm                communicator;

  LinePlotData<dim> data;

  // Global points
  std::vector<std::vector<Point<dim>>> global_points;

  bool cell_data_has_been_initialized;

  // For all lines: for all points along the line: for all relevant cells: dof index of first dof of
  // current cell and all shape function values
  std::vector<std::vector<std::vector<std::pair<unsigned int, std::vector<double>>>>>
    cells_global_velocity;

  // For all lines: for all points along the line: for all relevant cells: dof index of first dof of
  // current cell and all shape function values
  std::vector<std::vector<std::vector<std::pair<unsigned int, std::vector<double>>>>>
    cells_global_pressure;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // Velocity quantities
  // For all lines: for all points along the line
  std::vector<std::vector<Tensor<1, dim, double>>> velocity_global;

  // Pressure quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> pressure_global;

  bool write_final_output;
};



/*
 * This function calculates statistics along lines over time
 * and one spatial, homogeneous direction (averaging_direction = {0,1,2}), e.g.,
 * in the x-direction with a line in the y-z plane.
 *
 * NOTE: This function just works for geometries/meshes for which the cells are aligned with the
 * coordinate axis.
 */

// TODO Adapt code to geometries whose elements are not aligned with the coordinate axis.

template<int dim>
class LinePlotCalculatorStatisticsHomogeneousDirection
{
public:
  typedef LinearAlgebra::distributed::Vector<double> VectorType;

  typedef
    typename std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>>
      TYPE;

  LinePlotCalculatorStatisticsHomogeneousDirection(const DoFHandler<dim> & dof_handler_velocity_in,
                                                   const DoFHandler<dim> & dof_handler_pressure_in,
                                                   const Mapping<dim> &    mapping_in);

  void
  setup(LinePlotData<dim> const & line_statistics_data_in);

  void
  evaluate(VectorType const &   velocity,
           VectorType const &   pressure,
           double const &       time,
           unsigned int const & time_step_number);

private:
  void
  print_headline(std::ofstream & f, const unsigned int number_of_samples) const
  {
    f << "number of samples: N = " << number_of_samples << std::endl;
  }

  void
  do_evaluate(VectorType const & velocity, VectorType const & pressure);

  void
  do_evaluate_velocity(VectorType const & velocity,
                       Line<dim> const &  line,
                       unsigned int const line_iterator);

  void
  do_evaluate_pressure(VectorType const & pressure,
                       Line<dim> const &  line,
                       unsigned int const line_iterator);

  void
  average_pressure_for_given_point(VectorType const & pressure,
                                   TYPE const &       vector_cells_and_ref_points,
                                   double &           length_local,
                                   double &           pressure_local);

  void
  find_points_and_weights(Point<dim> const &        point_in_ref_coord,
                          std::vector<Point<dim>> & points,
                          std::vector<double> &     weights,
                          unsigned int const        averaging_direction,
                          QGauss<1> const &         gauss_1d);

  void
  do_write_output() const;

  mutable bool clear_files;

  DoFHandler<dim> const & dof_handler_velocity;
  DoFHandler<dim> const & dof_handler_pressure;
  Mapping<dim> const &    mapping;
  MPI_Comm                communicator;

  LinePlotData<dim> data;

  // Global points
  std::vector<std::vector<Point<dim>>> global_points;

  // For all lines: for all points along the line: list of all relevant cells and points in ref
  // coordinates
  std::vector<
    std::vector<std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>>>>
    cells_and_ref_points_velocity;

  // For all lines: for all points along the line: list of all relevant cells and points in ref
  // coordinates
  std::vector<
    std::vector<std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>>>>
    cells_and_ref_points_pressure;

  // For all lines: for pressure reference point: list of all relevant cells and points in ref
  // coordinates
  std::vector<std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>>>
    ref_pressure_cells_and_ref_points;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // homogeneous direction for averaging in space
  unsigned int averaging_direction;

  // Velocity quantities
  // For all lines: for all points along the line
  std::vector<std::vector<Tensor<1, dim, double>>> velocity_global;

  // Skin Friction quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> wall_shear_global;

  // Reynolds Stress quantities
  // For all lines: for all points along the line
  std::vector<std::vector<Tensor<2, dim, double>>> reynolds_global;

  // Pressure quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> pressure_global;
  // For all lines
  std::vector<double> reference_pressure_global;

  // write final output
  bool write_final_output;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_ \
        */
