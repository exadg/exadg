/*
 * stress_data.h
 *
 *  Created on: Aug 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_


#include <deal.II/base/types.h>

using namespace dealii;

enum class QuantityType
{
  Undefined,
  Velocity,
  Pressure
};

struct Quantity
{
  QuantityType type;

  // Additional data
};

template<int dim>
struct Line
{
  Line()
   :
   n_points(2),
   name("line")
  {}

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
  std::vector<Quantity> quantities;
};


template <int dim>
struct LinePlotData
{
  LinePlotData()
    :
    write_output(false),
    filename_prefix("output/"),
    precision(10)

  {}

  /*
   *  specify whether output is written or not
   */
  bool write_output;

  /*
   *  filename prefix (output folder)
   */
  std::string filename_prefix;

  /*
   *  precision (number of decimal places)
   */
  unsigned int precision;

  /*
   *  a vector of lines along which we want to write output
   */
  std::vector<Line<dim> > lines;
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_ */
