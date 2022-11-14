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

// ExaDG
#include <exadg/postprocessor/time_control_statistics.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
TimeControlDataStatistics::TimeControlDataStatistics()
  : write_preliminary_results_every_nth_time_step(numbers::invalid_timestep)
{
}

void
TimeControlDataStatistics::print(dealii::ConditionalOStream & pcout, bool const unsteady) const
{
  time_control_data.print(pcout, unsteady);
  if(Utilities::is_valid_timestep(write_preliminary_results_every_nth_time_step))
    print_parameter(pcout,
                    "Write preliminary results every nth time step",
                    write_preliminary_results_every_nth_time_step);
}

TimeControlStatistics::TimeControlStatistics() : final_output_written(false)
{
}

void
TimeControlStatistics::setup(TimeControlDataStatistics const & time_control_data_statistics_in)
{
  time_control_data_statistics = time_control_data_statistics_in;
  time_control.setup(time_control_data_statistics.time_control_data);
}

bool
TimeControlStatistics::write_preliminary_results(double const           time,
                                                 types::time_step const time_step_number) const
{
  if(Utilities::is_valid_timestep(
       time_control_data_statistics.write_preliminary_results_every_nth_time_step))
  {
    // if we are outside the interval bounds
    if(time <
         time_control_data_statistics.time_control_data.start_time - time_control.get_epsilon() ||
       final_output_written)
    {
      return false;
    }
    // check whether one of the criteria for writing results is met
    else if(time_control.reached_end_time() ||
            (time_step_number - 1) %
                (time_control_data_statistics.write_preliminary_results_every_nth_time_step) ==
              0)
    {
      // make sure that we do no longer write results once the end time has been reached
      if(time_control.reached_end_time())
        final_output_written = true;

      return true;
    }
  }

  return false;
}

} // namespace ExaDG
