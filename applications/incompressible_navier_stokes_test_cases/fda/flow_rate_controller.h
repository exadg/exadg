/*
 * flow_rate_controller.h
 *
 *  Created on: 30.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_FLOW_RATE_CONTROLLER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_FLOW_RATE_CONTROLLER_H_

namespace IncNS
{
namespace FDA
{
class FlowRateController
{
public:
  FlowRateController(double const target_flow_rate,
                     double const viscosity,
                     double const max_velocity,
                     double const R_outer,
                     double const mean_velocity_inflow,
                     double const D,
                     double const start_time)
    : target_flow_rate(target_flow_rate),
      // initialize the body force such that the desired flow rate is obtained
      // under the assumption of a parabolic velocity profile in radial direction
      f(4.0 * viscosity * max_velocity / std::pow(R_outer, 2.0)), // f(t=t_0) = f_0
      mean_velocity_inflow(mean_velocity_inflow),
      D(D),
      time_step(1.0),
      time_old(start_time),
      flow_rate(0.0)
  {
  }

  double
  get_body_force() const
  {
    return f;
  }

  void
  update_body_force(double const flow_rate_, double const time_)
  {
    flow_rate = flow_rate_;
    time_step = time_ - time_old;

    // use an I-controller to asymptotically reach the desired target flow rate

    // dimensional analysis: [k] = 1/(m^2 s^2) -> k = const * U_{mean,inflow}^2 / D^4
    // constant: choose a default value of 1
    double const k = 1.0e0 * std::pow(mean_velocity_inflow, 2.0) / std::pow(D, 4.0);
    f += k * (target_flow_rate - flow_rate) * time_step;

    time_old = time_;
  }

private:
  double target_flow_rate;

  double f;

  double const mean_velocity_inflow;
  double const D;

  double time_step;
  double time_old;

  double flow_rate;
};

} // namespace FDA
} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_FLOW_RATE_CONTROLLER_H_ */
