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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_FLOW_RATE_CONTROLLER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_FLOW_RATE_CONTROLLER_H_

namespace ExaDG
{
namespace IncNS
{
class FlowRateController
{
public:
  FlowRateController(double const bulk_velocity,
                     double const target_flow_rate,
                     double const H,
                     double const start_time)
    : bulk_velocity(bulk_velocity),
      target_flow_rate(target_flow_rate),
      length_scale(H),
      f(0.0), // f(t=t_0) = f_0
      f_damping(0.0),
      time_step(1.0),
      time_old(start_time),
      flow_rate(0.0),
      flow_rate_old(0.0)
  {
  }

  double
  get_body_force() const
  {
    return f + f_damping;
  }

  void
  update_body_force(double const       flow_rate_in,
                    double const       time,
                    unsigned int const time_step_number)
  {
    flow_rate = flow_rate_in;
    time_step = time - time_old;

    // use an I-controller with damping (D) to asymptotically reach the desired target flow rate

    // dimensional analysis: [k_I] = 1/(m^2 s^2) -> k_I = const * u_b^2 / H^4
    double const C_I = 100.0;
    double const k_I = C_I * std::pow(bulk_velocity, 2.0) / std::pow(length_scale, 4.0);
    f += k_I * (target_flow_rate - flow_rate) * time_step;

    // the time step size is 0 when this function is called the first time
    if(time_step_number > 1)
    {
      // dimensional analysis: [k_D] = 1/(m^2) -> k_D = const / H^2
      double const C_D = 0.1;
      double const k_D = C_D / std::pow(length_scale, 2.0);
      f_damping        = -k_D * (flow_rate - flow_rate_old) / time_step;
    }

    flow_rate_old = flow_rate;
    time_old      = time;
  }

private:
  double const bulk_velocity, target_flow_rate, length_scale;

  double f;
  double f_damping;

  double time_step;
  double time_old;

  double flow_rate;
  double flow_rate_old;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_FLOW_RATE_CONTROLLER_H_ \
        */
