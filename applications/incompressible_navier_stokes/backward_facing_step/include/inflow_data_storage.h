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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_INFLOW_DATA_STORAGE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_INFLOW_DATA_STORAGE_H_

// backward facing step application
#include "geometry.h"

namespace ExaDG
{
namespace IncNS
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

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_INFLOW_DATA_STORAGE_H_ \
        */
