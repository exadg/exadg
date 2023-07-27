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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_STEP_CALCULATION_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_STEP_CALCULATION_H_

namespace ExaDG
{
/*
 *  limit the maximum increase/decrease of the time step size
 */
inline void
limit_time_step_change(double & new_time_step, double const & last_time_step, double const & fac)
{
  if(new_time_step >= fac * last_time_step)
  {
    new_time_step = fac * last_time_step;
  }
  else if(new_time_step <= last_time_step / fac)
  {
    new_time_step = last_time_step / fac;
  }
}

/*
 * Decrease time_step in order to exactly hit end_time.
 */
inline double
adjust_time_step_to_hit_end_time(double const start_time,
                                 double const end_time,
                                 double const time_step)
{
  return (end_time - start_time) / (1 + int((end_time - start_time) / time_step));
}

/*
 * This function calculates the time step size for a given time step size and a specified number of
 * refinements, where the time step size is reduced by a factor of 2 for each refinement level.
 */
inline double
calculate_const_time_step(double const dt, unsigned int const n_refine_time)
{
  double const time_step = dt / std::pow(2., n_refine_time);

  return time_step;
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_STEP_CALCULATION_H_ */
