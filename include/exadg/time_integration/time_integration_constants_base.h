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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_INTEGRATION_CONSTANTS_BASE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_INTEGRATION_CONSTANTS_BASE_H_


namespace ExaDG
{
class TimeIntegratorConstantsBase
{
public:
  TimeIntegratorConstantsBase(unsigned int const order, bool const start_with_low_order)
    : order(order), start_with_low_order(start_with_low_order)
  {
  }

  virtual ~TimeIntegratorConstantsBase()
  {
  }

  /*
   *  This function updates the time integrator constants in case of constant time step sizes.
   */
  void
  update(unsigned int const current_order)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(current_order <= order && start_with_low_order == true)
    {
      set_constant_time_step(current_order);
    }
    else
    {
      set_constant_time_step(order);
    }
  }

  /*
   *  This function updates the time integrator constants in case of adaptive time step sizes.
   */
  void
  update(unsigned int const current_order, std::vector<double> const & time_steps)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(current_order <= order && start_with_low_order == true)
    {
      set_adaptive_time_step(current_order, time_steps);
    }
    else // adjust time integrator constants since this is adaptive time stepping
    {
      set_adaptive_time_step(order, time_steps);
    }
  }

  unsigned int
  get_order() const
  {
    return order;
  }

  /*
   *  This function prints the time integrator constants
   */
  virtual void
  print(dealii::ConditionalOStream & pcout) const = 0;

protected:
  // order of time integrator
  unsigned int const order;

  // use a low order time integration scheme to start the time integrator?
  bool const start_with_low_order;

private:
  /*
   *  This function calculates the time integrator constants in case of constant time step sizes.
   */
  virtual void
  set_constant_time_step(unsigned int const current_order) = 0;

  /*
   *  This function calculates time integrator constants in case of varying time step sizes
   * (adaptive time stepping).
   */
  virtual void
  set_adaptive_time_step(unsigned int const          current_order,
                         std::vector<double> const & time_steps) = 0;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_INTEGRATION_CONSTANTS_BASE_H_ */
