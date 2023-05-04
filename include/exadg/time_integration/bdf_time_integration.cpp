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
#include <exadg/time_integration/bdf_time_integration.h>

namespace ExaDG
{
BDFTimeIntegratorConstants::BDFTimeIntegratorConstants(unsigned int const order,
                                                       bool const         start_with_low_order)
  : TimeIntegratorConstantsBase(order, start_with_low_order), gamma0(-1.0), alpha(order)
{
  AssertThrow(order >= 1 and order <= 4,
              dealii::ExcMessage("Specified order of BDF scheme not implemented."));

  // The default case is start_with_low_order = false.
  set_constant_time_step(order);
}

double
BDFTimeIntegratorConstants::get_gamma0() const
{
  AssertThrow(gamma0 > 0.0, dealii::ExcMessage("Constant gamma0 has not been initialized."));

  return gamma0;
}

double
BDFTimeIntegratorConstants::get_alpha(unsigned int const i) const
{
  AssertThrow(i < order,
              dealii::ExcMessage(
                "In order to access BDF time integrator constants, the index "
                "has to be smaller than the order of the time integration scheme."));

  return alpha[i];
}


void
BDFTimeIntegratorConstants::set_constant_time_step(unsigned int const current_order)
{
  AssertThrow(current_order <= order,
              dealii::ExcMessage(
                "There is a logical error when updating the BDF time integrator constants."));

  if(current_order == 1) // BDF 1
  {
    gamma0 = 1.0;

    alpha[0] = 1.0;
  }
  else if(current_order == 2) // BDF 2
  {
    gamma0 = 3.0 / 2.0;

    alpha[0] = 2.0;
    alpha[1] = -0.5;
  }
  else if(current_order == 3) // BDF 3
  {
    gamma0 = 11. / 6.;

    alpha[0] = 3.;
    alpha[1] = -1.5;
    alpha[2] = 1. / 3.;
  }
  else if(current_order == 4) // BDF 4
  {
    gamma0 = 25. / 12.;

    alpha[0] = 4.;
    alpha[1] = -3.;
    alpha[2] = 4. / 3.;
    alpha[3] = -1. / 4.;
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g., when using start_with_low_order = true
   */
  for(unsigned int i = current_order; i < order; ++i)
  {
    alpha[i] = 0.0;
  }
}


void
BDFTimeIntegratorConstants::set_adaptive_time_step(unsigned int const          current_order,
                                                   std::vector<double> const & time_steps)
{
  AssertThrow(current_order <= order,
              dealii::ExcMessage(
                "There is a logical error when updating the time integrator constants."));

  AssertThrow(
    time_steps.size() == order,
    dealii::ExcMessage(
      "Length of vector containing time step sizes has to be equal to order of time integration scheme."));

  if(current_order == 1) // BDF 1
  {
    gamma0 = 1.0;

    alpha[0] = 1.0;
  }
  else if(current_order == 2) // BDF 2
  {
    gamma0 = (2 * time_steps[0] + time_steps[1]) / (time_steps[0] + time_steps[1]);

    alpha[0] = (time_steps[0] + time_steps[1]) / time_steps[1];
    alpha[1] = -time_steps[0] * time_steps[0] / ((time_steps[0] + time_steps[1]) * time_steps[1]);
  }
  else if(current_order == 3) // BDF 3
  {
    gamma0 = 1.0 + time_steps[0] / (time_steps[0] + time_steps[1]) +
             time_steps[0] / (time_steps[0] + time_steps[1] + time_steps[2]);

    alpha[0] = +(time_steps[0] + time_steps[1]) * (time_steps[0] + time_steps[1] + time_steps[2]) /
               (time_steps[1] * (time_steps[1] + time_steps[2]));
    alpha[1] = -time_steps[0] * time_steps[0] * (time_steps[0] + time_steps[1] + time_steps[2]) /
               ((time_steps[0] + time_steps[1]) * time_steps[1] * time_steps[2]);
    alpha[2] = +time_steps[0] * time_steps[0] * (time_steps[0] + time_steps[1]) /
               ((time_steps[0] + time_steps[1] + time_steps[2]) * (time_steps[1] + time_steps[2]) *
                time_steps[2]);
  }
  else if(current_order == 4) // BDF 4
  {
    gamma0 = 1.0 + time_steps[0] / (time_steps[0] + time_steps[1]) +
             time_steps[0] / (time_steps[0] + time_steps[1] + time_steps[2]) +
             time_steps[0] / (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]);

    alpha[0] = (time_steps[0] + time_steps[1]) * (time_steps[0] + time_steps[1] + time_steps[2]) *
               (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) /
               (time_steps[1] * (time_steps[1] + time_steps[2]) *
                (time_steps[1] + time_steps[2] + time_steps[3]));

    alpha[1] = -time_steps[0] * time_steps[0] * (time_steps[0] + time_steps[1] + time_steps[2]) *
               (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) /
               ((time_steps[0] + time_steps[1]) * time_steps[1] * time_steps[2] *
                (time_steps[2] + time_steps[3]));

    alpha[2] = time_steps[0] * time_steps[0] * (time_steps[0] + time_steps[1]) *
               (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) /
               ((time_steps[0] + time_steps[1] + time_steps[2]) * (time_steps[1] + time_steps[2]) *
                time_steps[2] * time_steps[3]);

    alpha[3] = -time_steps[0] * time_steps[0] * (time_steps[0] + time_steps[1]) *
               (time_steps[0] + time_steps[1] + time_steps[2]) /
               ((time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) *
                (time_steps[1] + time_steps[2] + time_steps[3]) * (time_steps[2] + time_steps[3]) *
                time_steps[3]);
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true
   */
  for(unsigned int i = current_order; i < order; ++i)
  {
    alpha[i] = 0.0;
  }
}

void
BDFTimeIntegratorConstants::print(dealii::ConditionalOStream & pcout) const
{
  pcout << "Gamma0   = " << gamma0 << std::endl;

  for(unsigned int i = 0; i < order; ++i)
    pcout << "Alpha[" << i << "] = " << alpha[i] << std::endl;
}

} // namespace ExaDG
