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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/time_integration/adams_bashforth_time_integration.h>

namespace ExaDG
{
AdamsBashforthTimeIntegratorConstants::AdamsBashforthTimeIntegratorConstants(
  unsigned int const order_time_integrator,
  bool const         start_with_low_order_method)
  : order(order_time_integrator), start_with_low_order(start_with_low_order_method), alpha(order)
{
  // In case of  ABM it is possible to have AB of order 0
  AssertThrow(order <= 4,
              dealii::ExcMessage("Specified order of Adams-Bashforth scheme not implemented."));

  // The default case is start_with_low_order = false.
  set_constant_time_step(order);
}

double
AdamsBashforthTimeIntegratorConstants::get_alpha(unsigned int const i) const
{
  AssertThrow(i < order,
              dealii::ExcMessage(
                "In order to access time integrator constants, the index "
                "has to be smaller than the order of the time integration scheme."));

  return alpha[i];
}


void
AdamsBashforthTimeIntegratorConstants::set_constant_time_step(unsigned int const current_order)
{
  AssertThrow(current_order <= order,
              dealii::ExcMessage(
                "There is a logical error when updating the AB time integrator constants."));

  if(current_order == 1) // AB 1
  {
    alpha[0] = 1.0;
  }
  else if(current_order == 2) // AB 2
  {
    alpha[0] = 3.0 / 2.0;
    alpha[1] = -1.0 / 2.0;
  }
  else if(current_order == 3) // AB 3
  {
    alpha[0] = 23.0 / 12.0;
    alpha[1] = -16.0 / 12.0;
    alpha[2] = 5.0 / 12.0;
  }
  else if(current_order == 4) // AB 4
  {
    alpha[0] = 55.0 / 24.0;
    alpha[1] = -59.0 / 24.0;
    alpha[2] = 37.0 / 24.0;
    alpha[3] = -9.0 / 24.0;
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g., when using start_with_low_order = true
   */
  for(unsigned int i = current_order; i < alpha.size(); ++i)
  {
    alpha[i] = 0.0;
  }
}


void
AdamsBashforthTimeIntegratorConstants::set_adaptive_time_step(
  unsigned int const          current_order,
  std::vector<double> const & time_steps)
{
  if(current_order == 1) // AB 1
  {
    alpha[0] = 1.0;
  }
  else if(current_order == 2) // AB 2
  {
    alpha[0] = (1.0 / 2.0 * time_steps[0] + time_steps[1]) / (time_steps[1]);
    alpha[1] = time_steps[0] / (-2.0 * time_steps[1]);
  }
  else if(current_order == 3) // AB 3
  {
    alpha[0] = 1.0 + (1.0 / 3.0 * time_steps[0] * time_steps[0] +
                      1.0 / 2.0 * time_steps[0] * (2.0 * time_steps[1] + time_steps[2])) /
                       (time_steps[1] * (time_steps[1] + time_steps[2]));

    alpha[1] = (1.0 / 3.0 * time_steps[0] * time_steps[0] +
                1.0 / 2.0 * time_steps[0] * (time_steps[1] + time_steps[2])) /
               (-time_steps[1] * time_steps[2]);

    alpha[2] =
      (1.0 / 3.0 * time_steps[0] * time_steps[0] + 1.0 / 2.0 * time_steps[0] * time_steps[1]) /
      (time_steps[2] * (time_steps[1] + time_steps[2]));
  }
  else if(current_order == 4) // AB 4
  {
    alpha[0] = 1.0 + (1.0 / 4.0 * time_steps[0] * time_steps[0] * time_steps[0] +
                      1.0 / 3.0 * time_steps[0] * time_steps[0] *
                        (3 * time_steps[1] + 2.0 * time_steps[2] + time_steps[3]) +
                      1.0 / 2.0 * time_steps[0] *
                        (3.0 * time_steps[1] * time_steps[1] + 2.0 * time_steps[1] * time_steps[3] +
                         time_steps[2] * time_steps[2] + time_steps[2] * time_steps[3] +
                         4.0 * time_steps[1] * time_steps[2])) /
                       (time_steps[1] * (time_steps[1] + time_steps[2]) *
                        (time_steps[1] + time_steps[2] + time_steps[3]));

    alpha[1] = (1.0 / 4.0 * time_steps[0] * time_steps[0] * time_steps[0] +
                1.0 / 3.0 * time_steps[0] * time_steps[0] *
                  (2.0 * time_steps[1] + 2.0 * time_steps[2] + time_steps[3]) +
                1.0 / 2.0 * time_steps[0] * (time_steps[1] + time_steps[2]) *
                  (time_steps[1] + time_steps[2] + time_steps[3])) /
               (-time_steps[1] * time_steps[2] * (time_steps[2] + time_steps[3]));

    alpha[2] = (1.0 / 4.0 * time_steps[0] * time_steps[0] * time_steps[0] +
                1.0 / 3.0 * time_steps[0] * time_steps[0] *
                  (2.0 * time_steps[1] + time_steps[2] + time_steps[3]) +
                1.0 / 2.0 * time_steps[0] * time_steps[1] *
                  (time_steps[1] + time_steps[2] + time_steps[3])) /
               (time_steps[3] * time_steps[2] * (time_steps[1] + time_steps[2]));

    alpha[3] = (1.0 / 4.0 * time_steps[0] * time_steps[0] * time_steps[0] +
                1.0 / 3.0 * time_steps[0] * time_steps[0] * (2.0 * time_steps[1] + time_steps[2]) +
                1.0 / 2.0 * time_steps[0] * time_steps[1] * (time_steps[1] + time_steps[2])) /
               (-time_steps[3] * (time_steps[2] + time_steps[3]) *
                (time_steps[1] + time_steps[2] + time_steps[3]));
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true
   */
  for(unsigned int i = current_order; i < alpha.size(); ++i)
  {
    alpha[i] = 0.0;
  }
}

void
AdamsBashforthTimeIntegratorConstants::update(unsigned int const          current_order,
                                              std::vector<double> const & time_steps,
                                              bool const                  adaptive)
{
  // when starting the time integrator with a low order method, ensure that
  // the time integrator constants are set properly
  unsigned int update_order =
    (current_order <= order && start_with_low_order == true) ? current_order : order;

  if(adaptive)
    set_adaptive_time_step(update_order, time_steps);
  else
    set_constant_time_step(update_order);
}

void
AdamsBashforthTimeIntegratorConstants::print(dealii::ConditionalOStream & pcout) const
{
  for(unsigned int i = 0; i < alpha.size(); ++i)
    pcout << "Alpha[" << i << "] = " << alpha[i] << std::endl;
}

} // namespace ExaDG
