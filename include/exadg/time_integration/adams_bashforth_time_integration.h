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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_ADAMS_BASHFORTH_TIME_INTEGRATION_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_ADAMS_BASHFORTH_TIME_INTEGRATION_H_

// C/C++
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>

namespace ExaDG
{
class AdamsBashforthTimeIntegratorConstants
{
public:
  AdamsBashforthTimeIntegratorConstants(unsigned int const order_time_integrator,
                                        bool const         start_with_low_order_method);

  double
  get_alpha(unsigned int const i) const;

  /*
   *  This function updates the time integrator constants of the AB scheme
   */
  void
  update(unsigned int const          time_step_number,
         std::vector<double> const & time_steps,
         bool const                  adaptive);

  /*
   *  This function prints the time integrator constants
   */
  void
  print(dealii::ConditionalOStream & pcout) const;


private:
  /*
   *  This function calculates the time integrator constants of the AB scheme
   *  in case of constant time step sizes.
   */
  void
  set_constant_time_step(unsigned int const current_order);

  /*
   *  This function calculates time integrator constants
   *  in case of varying time step sizes (adaptive time stepping).
   */
  void
  set_adaptive_time_step(unsigned int const current_order, std::vector<double> const & time_steps);

  // order of time integrator
  unsigned int const order;

  // use a low order time integration scheme to start the time integrator?
  bool const start_with_low_order;

  /*
   *  AB time integrator constants:
   *
   *  du/dt = (alpha_0 f^{n+1} + alpha_1 f^{n} + alpha_2 f^{n-1} +...)
   */
  std::vector<double> alpha;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_ADAMS_BASHFORTH_TIME_INTEGRATION_H_ */
