/*
 * line_plot_data.h
 *
 *  Created on: Aug 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_


#include <deal.II/base/point.h>

#include "../../functionalities/print_functions.h"

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
  LinePlotData() : calculate(false), directory("output/"), precision(10)

  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    // TODO: add output for basic line plot data

    if(statistics_data.calculate_statistics == true)
    {
      statistics_data.print(pcout);
    }
  }

  /*
   *  activates line plot calculation
   */
  bool calculate;

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

  /*
   *  Statistics data (only relevant if statistics have to
   *  evaluated but not for instantaneous results)
   */
  StatisticsData statistics_data;
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_ */
