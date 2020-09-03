/*
 * inflow_data_storage.h
 *
 *  Created on: 30.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_INFLOW_DATA_STORAGE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_INFLOW_DATA_STORAGE_H_

#include "geometry.h"

namespace ExaDG
{
namespace IncNS
{
namespace BackwardFacingStep
{
using namespace dealii;

template<int dim>
struct InflowDataStorage
{
  InflowDataStorage(unsigned int const n_points) : n_points_y(n_points), n_points_z(n_points)
  {
    y_values.resize(n_points_y);
    z_values.resize(n_points_z);
    velocity_values.resize(n_points_y * n_points_z);

    initialize_y_and_z_values();
    initialize_velocity_values();
  }

  void
  initialize_y_and_z_values()
  {
    AssertThrow(n_points_y >= 2, ExcMessage("Variable n_points_y is invalid"));
    AssertThrow(n_points_z >= 2, ExcMessage("Variable n_points_z is invalid"));

    for(unsigned int i = 0; i < n_points_y; ++i)
      y_values[i] = double(i) / double(n_points_y - 1) * Geometry::HEIGHT_CHANNEL;

    for(unsigned int i = 0; i < n_points_z; ++i)
      z_values[i] = -Geometry::WIDTH_CHANNEL / 2.0 +
                    double(i) / double(n_points_z - 1) * Geometry::WIDTH_CHANNEL;
  }

  void
  initialize_velocity_values()
  {
    AssertThrow(n_points_y >= 2, ExcMessage("Variable n_points_y is invalid"));
    AssertThrow(n_points_z >= 2, ExcMessage("Variable n_points_z is invalid"));

    for(unsigned int iy = 0; iy < n_points_y; ++iy)
    {
      for(unsigned int iz = 0; iz < n_points_z; ++iz)
      {
        Tensor<1, dim, double> velocity;
        velocity_values[iy * n_points_z + iz] = velocity;
      }
    }
  }

  unsigned int n_points_y;
  unsigned int n_points_z;

  std::vector<double>                 y_values;
  std::vector<double>                 z_values;
  std::vector<Tensor<1, dim, double>> velocity_values;
};

} // namespace BackwardFacingStep
} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_INFLOW_DATA_STORAGE_H_ \
        */
