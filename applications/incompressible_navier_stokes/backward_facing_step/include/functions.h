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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_BACKWARD_FACING_STEP_INCLUDE_FUNCTIONS_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_BACKWARD_FACING_STEP_INCLUDE_FUNCTIONS_H_

#include "inflow_data_storage.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
class InitialSolutionVelocity : public dealii::Function<dim>
{
public:
  InitialSolutionVelocity(double const max_velocity,
                          double const length,
                          double const height,
                          double const width)
    : dealii::Function<dim>(dim, 0.0),
      max_velocity(max_velocity),
      length(length),
      height(height),
      width(width)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    AssertThrow(dim == 3, dealii::ExcMessage("Dimension has to be dim==3."));

    double const x = p[0] / length;
    double const y = (p[1] - height / 2.0) / (height / 2.0);
    double const z = p[2] / width;

    double const factor = 0.5;

    double result = 0.0;
    if(false) // only random perturbations
    {
      if(component == 0)
      {
        if(std::abs(y) < 1.0)
          result = -max_velocity * (pow(y, 6.0) - 1.0) *
                   (1.0 + ((double)rand() / RAND_MAX - 0.5) * factor);
        else
          result = 0.0;
      }
    }
    else // random perturbations and vortices
    {
      if(component == 0)
      {
        if(std::abs(y) < 1.0)
          result = -max_velocity * (pow(y, 6.0) - 1.0) *
                   (1.0 + (((double)rand() / RAND_MAX - 1.0) + std::sin(z * 8.0) * 0.5) * factor);
      }

      if(component == 2)
      {
        if(std::abs(y) < 1.0)
          result = -max_velocity * (pow(y, 6.0) - 1.0) * std::sin(x * 8.0) * 0.5 * factor;
      }
    }

    return result;
  }

private:
  double const max_velocity, length, height, width;
};

template<int dim>
class InflowProfile : public dealii::Function<dim>
{
public:
  InflowProfile(InflowDataStorage<dim> const & inflow_data_storage)
    : dealii::Function<dim>(dim, 0.0), data(inflow_data_storage)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double result = linear_interpolation_2d_cartesian(
      p, data.y_values, data.z_values, data.velocity_values, component);

    return result;
  }

private:
  InflowDataStorage<dim> const & data;
};

template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide() : dealii::Function<dim>(dim, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & /*p*/, unsigned int const component = 0) const final
  {
    double result = 0.0;

    // channel flow with periodic boundary conditions.
    // The force is known (so no flow rate controller is used).
    if(component == 0)
      return 0.2844518;
    else
      return 0.0;

    return result;
  }
};

} // namespace IncNS
} // namespace ExaDG

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_BACKWARD_FACING_STEP_INCLUDE_FUNCTIONS_H_ */
