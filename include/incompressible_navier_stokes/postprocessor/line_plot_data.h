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
  Pressure,
  SkinFriction,
  ReynoldsStresses,
  PressureCoefficient
};

struct Quantity
{
  Quantity()
    :
    type(QuantityType::Undefined)
  {}

  Quantity(QuantityType const &quantity_type)
    :
    type(quantity_type)
  {}

  virtual ~Quantity(){}

  QuantityType type;
};

struct QuantityStatistics : Quantity
{
  QuantityStatistics()
    :
    Quantity(),
    averaging_direction(0)
  {}

  unsigned int averaging_direction; //x = 0, y = 1, z = 2
};

template<int dim>
struct QuantityStatisticsPressureCoefficient : QuantityStatistics
{
  QuantityStatisticsPressureCoefficient()
    :
    QuantityStatistics(),
    reference_velocity(0.0),
    reference_point(Point<dim>())
  {}

  double reference_velocity;
  Point<dim> reference_point;
};

template<int dim>
struct QuantityStatisticsSkinFriction : QuantityStatistics
{
  QuantityStatisticsSkinFriction()
    :
    QuantityStatistics(),
    reference_velocity(1.0),
    viscosity(1.0),
    normal_vector(Tensor<1,dim,double>()),
    tangent_vector(Tensor<1,dim,double>())
  {}

  double reference_velocity;
  double viscosity;
  Tensor<1,dim,double> normal_vector;
  Tensor<1,dim,double> tangent_vector;
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
  std::vector<Quantity*> quantities;
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
