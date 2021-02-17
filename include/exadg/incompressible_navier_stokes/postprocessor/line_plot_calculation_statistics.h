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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_

// C++
#include <fstream>

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_data.h>

namespace ExaDG
{
namespace IncNS
{
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

template<int dim, typename Number>
class LinePlotCalculatorStatistics
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef
    typename std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>>
      TYPE;

  LinePlotCalculatorStatistics(const DoFHandler<dim> & dof_handler_velocity_in,
                               const DoFHandler<dim> & dof_handler_pressure_in,
                               const Mapping<dim> &    mapping_in,
                               MPI_Comm const &        mpi_comm_in);

  void
  setup(LinePlotDataStatistics<dim> const & line_plot_data_in);

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
  initialize_cell_data(VectorType const & velocity, VectorType const & pressure);

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

  LinePlotDataStatistics<dim> data;

  // Global points
  std::vector<std::vector<Point<dim>>> global_points;

  bool cell_data_has_been_initialized;

  // For all lines: for all points along the line: for all relevant cells: dof index of first dof of
  // current cell and all shape function values
  std::vector<
    std::vector<std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>>>
    cells_global_velocity;

  // For all lines: for all points along the line: for all relevant cells: dof index of first dof of
  // current cell and all shape function values
  std::vector<
    std::vector<std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>>>
    cells_global_pressure;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // Velocity quantities
  // For all lines: for all points along the line
  std::vector<std::vector<Tensor<1, dim, Number>>> velocity_global;

  // Pressure quantities
  // For all lines: for all points along the line
  std::vector<std::vector<Number>> pressure_global;

  bool write_final_output;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_ \
        */
