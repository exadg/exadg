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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_HOMOGENEOUS_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_HOMOGENEOUS_H_

// C++
#include <fstream>

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_data.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * This function calculates statistics along lines over time
 * and one spatial, homogeneous direction (averaging_direction = {0,1,2}), e.g.,
 * in the x-direction with a line in the y-z plane.
 *
 * NOTE: This functionality can only be used for hypercube meshes and for geometries/meshes for
 * which the cells are aligned with the coordinate axis.
 */

template<int dim, typename Number>
class LinePlotCalculatorStatisticsHomogeneous
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>
    TYPE;

  LinePlotCalculatorStatisticsHomogeneous(dealii::DoFHandler<dim> const & dof_handler_velocity_in,
                                          dealii::DoFHandler<dim> const & dof_handler_pressure_in,
                                          dealii::Mapping<dim> const &    mapping_in,
                                          MPI_Comm const &                mpi_comm_in);

  void
  setup(LinePlotDataStatistics<dim> const & data_in);

  void
  evaluate(VectorType const & velocity, VectorType const & pressure);

  void
  write_output() const;

  TimeControlStatistics time_control_statistics;

private:
  void
  print_headline(std::ofstream & f, unsigned int const number_of_samples) const;

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
  find_points_and_weights(dealii::Point<dim> const &        point_in_ref_coord,
                          std::vector<dealii::Point<dim>> & points,
                          std::vector<double> &             weights,
                          unsigned int const                averaging_direction,
                          dealii::QGauss<1> const &         gauss_1d);

  void
  do_write_output() const;

  mutable bool clear_files;

  dealii::DoFHandler<dim> const & dof_handler_velocity;
  dealii::DoFHandler<dim> const & dof_handler_pressure;
  dealii::Mapping<dim> const &    mapping;
  MPI_Comm                        mpi_comm;

  LinePlotDataStatistics<dim> data;

  // Global points
  std::vector<std::vector<dealii::Point<dim>>> global_points;

  // For all lines: for all points along the line: list of all relevant cells and points in ref
  // coordinates
  std::vector<std::vector<std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>>>
    cells_and_ref_points_velocity;

  // For all lines: for all points along the line: list of all relevant cells and points in ref
  // coordinates
  std::vector<std::vector<std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>>>
    cells_and_ref_points_pressure;

  // For all lines: for pressure reference point: list of all relevant cells and points in ref
  // coordinates
  std::vector<std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>>
    cells_and_ref_points_ref_pressure;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // homogeneous direction for averaging in space
  unsigned int averaging_direction;

  // Velocity quantities
  // For all lines: for all points along the line
  std::vector<std::vector<dealii::Tensor<1, dim, double>>> velocity_global;

  // Skin Friction quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> wall_shear_global;

  // Reynolds Stress quantities
  // For all lines: for all points along the line
  std::vector<std::vector<dealii::Tensor<2, dim, double>>> reynolds_global;

  // Pressure quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> pressure_global;
  // For all lines
  std::vector<double> reference_pressure_global;

  // write final output
  bool write_final_output;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_HOMOGENEOUS_H_ \
        */
