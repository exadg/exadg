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

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TIME_CONTROL_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TIME_CONTROL_H_

/**
 *This class provides information at which time steps a certain action should be triggered. It is
 *typically used in postprocessing routines that shall not be invoked after every time step. The
 *user can specify an evaluation interval in terms of number of time steps or an amount of time.
 */

// deal.ii
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
struct TimeControlData
{
  TimeControlData();

  bool             is_active;
  double           start_time;
  double           end_time;
  double           trigger_interval;
  types::time_step trigger_every_time_steps;

  enum UnsteadyEvalType
  {
    None,
    Interval,
    Timestep
  };

  void
  print(dealii::ConditionalOStream & pcout, bool const unsteady) const;
};

TimeControlData::UnsteadyEvalType
get_unsteady_evaluation_type(TimeControlData const & data);

class TimeControl
{
public:
  TimeControl();

  void
  setup(TimeControlData const & time_control_data);

  bool
  needs_evaluation(double const time, types::time_step const time_step_number) const;

  unsigned int
  get_counter() const;

  bool
  reached_end_time() const;

  bool
  get_epsilon() const;

private:
  // small number which is much smaller than the time step size
  double const         EPSILON;
  mutable bool         reset_counter;
  mutable unsigned int counter;
  mutable bool         end_time_reached;

  TimeControlData time_control_data;
};


} // namespace ExaDG


#endif /*INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TIME_CONTROL_H_*/
