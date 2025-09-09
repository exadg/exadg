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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
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
  print(dealii::ConditionalOStream & pcout)
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
  std::vector<dealii::Tensor<1, dim, double>> * array;
};

template<int dim, typename Number>
class InflowDataCalculator
{
public:
  InflowDataCalculator(InflowData<dim> const & inflow_data, MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler_velocity, dealii::Mapping<dim> const & mapping);

  void
  calculate(dealii::LinearAlgebra::distributed::Vector<Number> const & velocity);

private:
  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler_velocity;
  dealii::ObserverPointer<dealii::Mapping<dim> const>    mapping;
  InflowData<dim>                                        inflow_data;
  bool                                                   inflow_data_has_been_initialized;

  MPI_Comm const mpi_comm;

  std::vector<
    std::vector<std::pair<std::vector<dealii::types::global_dof_index>, std::vector<Number>>>>
    array_dof_indices_and_shape_values;

  std::vector<unsigned int> array_counter;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_ */
