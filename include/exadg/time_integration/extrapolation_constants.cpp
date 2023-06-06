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

#ifndef INCLUDE_TIME_INTEGRATION_EXTRAPOLATION_SCHEME_CPP_
#define INCLUDE_TIME_INTEGRATION_EXTRAPOLATION_SCHEME_CPP_

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/time_integration/extrapolation_constants.h>

namespace ExaDG
{
ExtrapolationConstants::ExtrapolationConstants(unsigned int const order,
                                               bool const         start_with_low_order)
  : TimeIntegratorConstantsBase(order, start_with_low_order), beta(order)
{
  AssertThrow(order <= 4,
              dealii::ExcMessage("Specified order of extrapolation scheme not implemented."));

  // The default case is start_with_low_order = false.
  set_constant_time_step(order);
}

double
ExtrapolationConstants::get_beta(unsigned int const i) const
{
  AssertThrow(i < order,
              dealii::ExcMessage("In order to access constants of extrapolation scheme, the index "
                                 "has to be smaller than the order of the scheme."));

  return beta[i];
}

void
ExtrapolationConstants::set_constant_time_step(unsigned int const current_order)
{
  AssertThrow(
    current_order <= order,
    dealii::ExcMessage(
      "There is a logical error when updating the constants of the extrapolation scheme."));

  if(current_order == 1) // EX 1
  {
    beta[0] = 1.0;
  }
  else if(current_order == 2) // EX 2
  {
    beta[0] = 2.0;
    beta[1] = -1.0;
  }
  else if(current_order == 3) // EX 3
  {
    beta[0] = 3.0;
    beta[1] = -3.0;
    beta[2] = 1.0;
  }
  else if(current_order == 4) // EX 4
  {
    beta[0] = 4.;
    beta[1] = -6.;
    beta[2] = 4.;
    beta[3] = -1.;
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true
   */
  for(unsigned int i = current_order; i < order; ++i)
  {
    beta[i] = 0.0;
  }
}


void
ExtrapolationConstants::set_adaptive_time_step(unsigned int const          current_order,
                                               std::vector<double> const & time_steps)
{
  AssertThrow(
    current_order <= order,
    dealii::ExcMessage(
      "There is a logical error when updating the constants of the extrapolation scheme."));

  AssertThrow(
    order <= time_steps.size(),
    dealii::ExcMessage(
      "Length of vector containing time step sizes has to be equal to order of extrapolation scheme."));

  if(current_order == 1) // EX 1
  {
    beta[0] = 1.0;
  }
  else if(current_order == 2) // EX 2
  {
    beta[0] = (time_steps[0] + time_steps[1]) / time_steps[1];
    beta[1] = -time_steps[0] / time_steps[1];
  }
  else if(current_order == 3) // EX 3
  {
    beta[0] = +(time_steps[0] + time_steps[1]) * (time_steps[0] + time_steps[1] + time_steps[2]) /
              (time_steps[1] * (time_steps[1] + time_steps[2]));
    beta[1] = -time_steps[0] * (time_steps[0] + time_steps[1] + time_steps[2]) /
              (time_steps[1] * time_steps[2]);
    beta[2] = +time_steps[0] * (time_steps[0] + time_steps[1]) /
              ((time_steps[1] + time_steps[2]) * time_steps[2]);
  }
  else if(current_order == 4) // EX 4
  {
    beta[0] = (time_steps[0] + time_steps[1]) * (time_steps[0] + time_steps[1] + time_steps[2]) *
              (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) /
              (time_steps[1] * (time_steps[1] + time_steps[2]) *
               (time_steps[1] + time_steps[2] + time_steps[3]));

    beta[1] = -time_steps[0] * (time_steps[0] + time_steps[1] + time_steps[2]) *
              (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) /
              (time_steps[1] * time_steps[2] * (time_steps[2] + time_steps[3]));

    beta[2] = time_steps[0] * (time_steps[0] + time_steps[1]) *
              (time_steps[0] + time_steps[1] + time_steps[2] + time_steps[3]) /
              ((time_steps[1] + time_steps[2]) * time_steps[2] * time_steps[3]);

    beta[3] = -time_steps[0] * (time_steps[0] + time_steps[1]) *
              (time_steps[0] + time_steps[1] + time_steps[2]) /
              ((time_steps[1] + time_steps[2] + time_steps[3]) * (time_steps[2] + time_steps[3]) *
               time_steps[3]);
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true,
   * if current_order == 0, all coefficients are set to zero, i.e., no extrapolation
   */
  for(unsigned int i = current_order; i < order; ++i)
  {
    beta[i] = 0.0;
  }
}

void
ExtrapolationConstants::print(dealii::ConditionalOStream & pcout) const
{
  for(unsigned int i = 0; i < order; ++i)
    pcout << "Beta[" << i << "]  = " << beta[i] << std::endl;
}

} // namespace ExaDG

#endif /* INCLUDE_TIME_INTEGRATION_EXTRAPOLATION_SCHEME_CPP_ */
