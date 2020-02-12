/*
 * inflow_data_calculator.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../../functionalities/print_functions.h"

/*
 * inflow data: use velocity at the outflow of one domain as inflow-BC for another domain
 *
 * The outflow boundary has to be the y-z plane at a given x-coordinate. The velocity is written at
 * n_points_y in y-direction and n_points_z in z-direction, which has to be specified by the user.
 */
enum class InflowGeometry
{
  Cartesian,
  Cylindrical
};

template<int dim>
struct InflowData
{
  InflowData()
    : write_inflow_data(false),
      inflow_geometry(InflowGeometry::Cartesian),
      normal_direction(0),
      normal_coordinate(0.0),
      n_points_y(2),
      n_points_z(2),
      y_values(nullptr),
      z_values(nullptr),
      array(nullptr)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(write_inflow_data == true)
    {
      print_parameter(pcout, "Normal direction", normal_direction);
      print_parameter(pcout, "Normal coordinate", normal_coordinate);
      print_parameter(pcout, "Number of points in y-direction", n_points_y);
      print_parameter(pcout, "Number of points in z-direction", n_points_z);
    }
  }

  // write the data?
  bool write_inflow_data;

  InflowGeometry inflow_geometry;

  // direction normal to inflow surface: has to be 0 (x), 1 (y), or 2 (z)
  unsigned int normal_direction;
  // position of inflow plane in the direction normal to the inflow surface
  double normal_coordinate;
  // specify the number of data points (grid points) in y- and z-direction
  unsigned int n_points_y;
  unsigned int n_points_z;

  // Vectors with the y-coordinates, z-coordinates (in physical space)
  std::vector<double> * y_values;
  std::vector<double> * z_values;
  // and the velocity values at n_points_y*n_points_z points
  std::vector<Tensor<1, dim, double>> * array;
};

template<int dim, typename Number>
class InflowDataCalculator
{
public:
  InflowDataCalculator(InflowData<dim> const & inflow_data, MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const & dof_handler_velocity, Mapping<dim> const & mapping);

  void
  calculate(LinearAlgebra::distributed::Vector<Number> const & velocity);

private:
  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  SmartPointer<Mapping<dim> const>    mapping;
  InflowData<dim>                     inflow_data;
  bool                                inflow_data_has_been_initialized;

  MPI_Comm const & mpi_comm;

  std::vector<std::vector<std::pair<unsigned int, std::vector<Number>>>>
    array_dof_index_and_shape_values;

  std::vector<unsigned int> array_counter;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_ */
