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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_

// deal.II
#include <deal.II/base/point.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

enum class QuantityType
{
  Undefined,
  Velocity,
  Pressure,
  SkinFriction,
  ReynoldsStresses,
  PressureCoefficient
};

struct Quantity
{
  Quantity() : type(QuantityType::Undefined)
  {
  }

  Quantity(QuantityType const & quantity_type) : type(quantity_type)
  {
  }

  virtual ~Quantity()
  {
  }

  QuantityType type;
};

template<int dim>
struct QuantityPressureCoefficient : Quantity
{
  QuantityPressureCoefficient() : Quantity(), reference_point(Point<dim>())
  {
  }

  Point<dim> reference_point;
};

template<int dim>
struct QuantitySkinFriction : Quantity
{
  QuantitySkinFriction()
    : Quantity(),
      viscosity(1.0),
      normal_vector(Tensor<1, dim, double>()),
      tangent_vector(Tensor<1, dim, double>())
  {
  }

  double                 viscosity;
  Tensor<1, dim, double> normal_vector;
  Tensor<1, dim, double> tangent_vector;
};

template<int dim>
struct Line
{
  Line() : n_points(2), name("line")
  {
  }

  virtual ~Line()
  {
  }

  /*
   *  begin and end points of line
   */
  Point<dim> begin;
  Point<dim> end;

  /*
   *  number of data points written along a line
   */
  unsigned int n_points;

  /*
   *  name of line
   */
  std::string name;

  /*
   *  Specify for which fields/quantities to write output
   */
  std::vector<std::shared_ptr<Quantity>> quantities;
};

/*
 * Derived data structure in order to perform additional averaging
 * in circumferential direction for rotationally symmetric problems.
 *
 * Note that the implementation assumes that line->begin is the
 * center of the circle for circumferential averaging.
 */
template<int dim>
struct LineCircumferentialAveraging : Line<dim>
{
  LineCircumferentialAveraging()
    : Line<dim>(),
      average_circumferential(false),
      n_points_circumferential(4),
      normal_vector(Tensor<1, dim>())
  {
  }

  // activate averaging in circumferential direction
  bool average_circumferential;

  // number of points used for averaging in circumferential direction
  // only used in average_circumferential == true
  unsigned int n_points_circumferential;

  // defines the averaging plane along with the begin and end points
  // of the line. Has to be specified if averaging in circumferential
  // direction is activated.
  Tensor<1, dim> normal_vector;
};

/*
 * Derived data structure in order to perform additional averaging
 * in homogeneous direction.
 */
template<int dim>
struct LineHomogeneousAveraging : Line<dim>
{
  LineHomogeneousAveraging()
    : Line<dim>(), average_homogeneous_direction(false), averaging_direction(0)
  {
  }

  // activate averaging in homogeneous direction
  bool average_homogeneous_direction;

  // has to be specified in case of averaging in homogeneous direction
  unsigned int averaging_direction; // x = 0, y = 1, z = 2
};

struct StatisticsData
{
  StatisticsData()
    : calculate_statistics(false),
      sample_start_time(0.0),
      sample_end_time(1.0),
      sample_every_timesteps(1),
      write_output_every_timesteps(100)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate_statistics == true)
    {
      pcout << "  Statistics data:" << std::endl;
      print_parameter(pcout, "Calculate statistics", calculate_statistics);
      print_parameter(pcout, "Sample start time", sample_start_time);
      print_parameter(pcout, "Sample end time", sample_end_time);
      print_parameter(pcout, "Sample every timesteps", sample_every_timesteps);
      print_parameter(pcout, "Write output every timesteps", write_output_every_timesteps);
    }
  }

  // calculate statistics?
  bool calculate_statistics;

  // start time for sampling
  double sample_start_time;

  // end time for sampling
  double sample_end_time;

  // perform sampling every ... timesteps
  unsigned int sample_every_timesteps;

  // write output every ... timesteps
  unsigned int write_output_every_timesteps;
};


template<int dim>
struct LinePlotData
{
  LinePlotData() : directory("output/"), precision(10)

  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    // TODO: add output for basic line plot data
  }

  /*
   *  output folder
   */
  std::string directory;

  /*
   *  precision (number of decimal places when writing to files)
   */
  unsigned int precision;

  /*
   *  a vector of lines along which we want to write output
   */
  std::vector<std::shared_ptr<Line<dim>>> lines;
};

template<int dim>
struct LinePlotDataInstantaneous
{
  LinePlotDataInstantaneous() : calculate(false)

  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    line_data.print(pcout);
  }

  LinePlotData<dim> line_data;

  /*
   *  activates line plot calculation
   */
  bool calculate;
};

template<int dim>
struct LinePlotDataStatistics
{
  LinePlotDataStatistics()
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    line_data.print(pcout);

    if(statistics_data.calculate_statistics == true)
    {
      statistics_data.print(pcout);
    }
  }

  LinePlotData<dim> line_data;

  /*
   *  Statistics data (only relevant if statistics have to
   *  evaluated but not for instantaneous results)
   */
  StatisticsData statistics_data;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_ */
