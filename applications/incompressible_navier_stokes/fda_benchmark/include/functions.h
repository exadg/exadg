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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_FDA_BENCHMARK_INCLUDE_FUNCTIONS_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_FDA_BENCHMARK_INCLUDE_FUNCTIONS_H_

// FDA benchmark
#include "flow_rate_controller.h"
#include "grid.h"
#include "inflow_data_storage.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
class InitialSolutionVelocity : public dealii::Function<dim>
{
public:
  InitialSolutionVelocity(double const max_velocity)
    : dealii::Function<dim>(dim, 0.0), max_velocity(max_velocity)
  {
    srand(0); // initialize rand() to obtain reproducible results
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    AssertThrow(dim == 3, dealii::ExcMessage("Dimension has to be dim==3."));

    double result = 0.0;

    // flow in z-direction
    if(component == 2)
    {
      // assume parabolic profile u(r) = u_max * [1-(r/R)^2]
      //  -> u_max = 2 * u_mean = 2 * flow_rate / area
      double const R = FDANozzle::radius_function(p[2]);
      double const r = std::min(std::sqrt(p[0] * p[0] + p[1] * p[1]), R);

      // parabolic velocity profile
      double const max_velocity_z = max_velocity * std::pow(FDANozzle::R_OUTER / R, 2.0);

      result = max_velocity_z * (1.0 - pow(r / R, 2.0));

      // Add perturbation (sine + random) for the precursor to initiate
      // a turbulent flow in case the Reynolds number is large enough
      // (otherwise, the perturbations will be damped and the flow becomes laminar).
      // According to first numerical results, the perturbed flow returns to a laminar
      // steady state in the precursor domain for Reynolds numbers Re_t = 500, 2000,
      // 3500, 5000, and 6500.
      if(p[2] <= FDANozzle::Z2_PRECURSOR)
      {
        double const phi    = std::atan2(p[1], p[0]);
        double const factor = 0.5;

        double perturbation =
          factor * max_velocity_z * std::sin(4.0 * phi) *
            std::sin(8.0 * dealii::numbers::PI * p[2] / FDANozzle::LENGTH_PRECURSOR) +
          factor * max_velocity_z * ((double)rand() / RAND_MAX - 0.5) / 0.5;

        // the perturbations should fulfill the Dirichlet boundary conditions
        perturbation *= (1.0 - pow(r / R, 6.0));

        result += perturbation;
      }
    }

    return result;
  }

private:
  double const max_velocity;
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
    // compute polar coordinates (r, phi) from point p
    // given in Cartesian coordinates (x, y) = inflow plane
    double const r   = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    double const phi = std::atan2(p[1], p[0]);

    double const result = linear_interpolation_2d_cylindrical(
      r, phi, data.r_values, data.phi_values, data.velocity_values, component);

    return result;
  }

private:
  InflowDataStorage<dim> const & data;
};


/*
 *  Right-hand side function: Implements the body force vector occurring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations.
 *  Only relevant for precursor simulation.
 */
template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(FlowRateController const & flow_rate_controller)
    : dealii::Function<dim>(dim, 0.0), flow_rate_controller(flow_rate_controller)
  {
  }

  double
  value(dealii::Point<dim> const & /*p*/, unsigned int const component = 0) const final
  {
    double result = 0.0;

    // Channel flow with periodic BCs in z-direction:
    // The flow is driven by a body force in z-direction
    if(component == 2)
    {
      result = flow_rate_controller.get_body_force();
    }

    return result;
  }

private:
  FlowRateController const & flow_rate_controller;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_FDA_BENCHMARK_INCLUDE_FUNCTIONS_H_ */
