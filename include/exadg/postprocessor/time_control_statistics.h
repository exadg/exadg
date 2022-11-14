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

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TIME_CONTROL_STATISTICS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TIME_CONTROL_STATISTICS_H_

// ExaDG
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
struct TimeControlDataStatistics
{
  TimeControlDataStatistics();

  void
  print(dealii::ConditionalOStream & pcout, bool const unsteady) const;

  types::time_step write_preliminary_results_every_nth_time_step;

  TimeControlData time_control_data;
};


class TimeControlStatistics
{
public:
  TimeControlStatistics();

  void
  setup(TimeControlDataStatistics const & time_control_data_statistics_in);

  bool
  write_preliminary_results(double const time, types::time_step const time_step_number) const;

  TimeControl time_control;

private:
  TimeControlDataStatistics time_control_data_statistics;
  mutable bool              final_output_written;
};


} // namespace ExaDG

#endif /*INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TIME_CONTROL_STATISTICS_H_*/
