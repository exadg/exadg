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

#ifndef INCLUDE_FUNCTIONALITIES_SOLVER_INFO_DATA_H_
#define INCLUDE_FUNCTIONALITIES_SOLVER_INFO_DATA_H_

// C/C++
#include <limits>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
using namespace dealii;

struct SolverInfoData
{
  SolverInfoData()
    : interval_time(std::numeric_limits<double>::max()),
      interval_wall_time(std::numeric_limits<double>::max()),
      interval_time_steps(std::numeric_limits<unsigned int>::max()),
      counter(0),
      do_output_in_this_time_step(false),
      old_time_step_number(0)
  {
  }

  void
  print(ConditionalOStream const & pcout) const
  {
    pcout << "  Solver information:" << std::endl;
    print_parameter(pcout, "Interval physical time", interval_time);
    print_parameter(pcout, "Interval wall time", interval_wall_time);
    print_parameter(pcout, "Interval time steps", interval_time_steps);
  }

  bool
  check_for_output(double const       wall_time,
                   double const       time,
                   unsigned int const time_step_number) const
  {
    // After a restart, the counter is reset to 1, but time = current_time - start time != 0 after a
    // restart. Hence, we have to explicitly reset the counter in that case. There is nothing to do
    // if the restart is controlled by the wall time or the time_step_number because these
    // variables are reinitialized after a restart anyway.
    if(time_step_number == 1)
    {
      counter += int((time + 1.e-10) / interval_time);
    }

    do_output_in_this_time_step = wall_time > interval_wall_time * counter ||
                                  time > interval_time * counter ||
                                  time_step_number % interval_time_steps == 0;

    if(do_output_in_this_time_step)
    {
      ++counter;
    }

    return do_output_in_this_time_step;
  }

  bool
  write(double const wall_time, double const time, unsigned int const time_step_number) const
  {
    if(time_step_number > old_time_step_number)
    {
      old_time_step_number = time_step_number;
      return check_for_output(wall_time, time, time_step_number);
    }
    else
    {
      return do_output_in_this_time_step;
    }
  }

  // physical time
  double interval_time;

  // wall time in seconds (= hours * 3600)
  double interval_wall_time;

  // number of time steps after which to write restart
  unsigned int interval_time_steps;

  // counter needed do decide when to write restart
  mutable unsigned int counter;

  // variable that stores whether output should be printed in current time step
  mutable bool do_output_in_this_time_step;

  // we need to store the old time step number since the function write() might be called multiple
  // times during one time step
  mutable unsigned int old_time_step_number;
};

} // namespace ExaDG


#endif /* INCLUDE_FUNCTIONALITIES_SOLVER_INFO_DATA_H_ */
