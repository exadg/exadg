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

// C/C++
#include <limits>

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
TimeControlData::UnsteadyEvalType
get_unsteady_evaluation_type(TimeControlData const & data)
{
  bool const interval = (data.trigger_interval > 0.0);
  bool const timestep = Utilities::is_valid_timestep(data.trigger_every_time_steps);

  // We do not allow users to specify both a time interval and a number of time steps.
  if(interval and timestep)
    return TimeControlData::UnsteadyEvalType::None;

  if(interval)
    return TimeControlData::UnsteadyEvalType::Interval;

  if(timestep)
    return TimeControlData::UnsteadyEvalType::Timestep;

  return TimeControlData::UnsteadyEvalType::None;
}

TimeControlData::TimeControlData()
  : is_active(false),
    start_time(std::numeric_limits<double>::max()),
    end_time(std::numeric_limits<double>::max()),
    trigger_interval(-1.0),
    trigger_every_time_steps(numbers::invalid_timestep)
{
}

void
TimeControlData::print(dealii::ConditionalOStream & pcout, bool const unsteady) const
{
  if(unsteady or is_active)
  {
    print_parameter(pcout, "TimeControl start time", start_time);
    print_parameter(pcout, "TimeControl end_time", end_time);
    if(get_unsteady_evaluation_type(*this) == TimeControlData::UnsteadyEvalType::Interval)
      print_parameter(pcout, "TimeControl triggers every interval", trigger_interval);
    if(get_unsteady_evaluation_type(*this) == TimeControlData::UnsteadyEvalType::Timestep)
      print_parameter(pcout, "TimeControl triggers every timestep", trigger_every_time_steps);
  }
}

TimeControl::TimeControl()
  : EPSILON(1.0e-10), reset_counter(true), counter(0), end_time_reached(false)
{
}

void
TimeControl::setup(TimeControlData const & time_control_data_in)
{
  time_control_data = time_control_data_in;
}

bool
TimeControl::needs_evaluation(double const time, types::time_step const time_step_number) const
{
  if(time_control_data.is_active == false)
    return false;

  // steady case
  if(not Utilities::is_unsteady_timestep(time_step_number))
  {
    ++counter;
    return true;
  }

  // unsteady evaluation
  AssertThrow(get_unsteady_evaluation_type(time_control_data) !=
                TimeControlData::UnsteadyEvalType::None,
              dealii::ExcMessage(
                "Unsteady evaluation not possible since it is not specified when to evaluate."));

  if(time < time_control_data.start_time - EPSILON)
    return false;

  if(time > time_control_data.end_time - EPSILON)
    end_time_reached = true;

  if(time > time_control_data.end_time + EPSILON)
    return false;

  bool evaluate = false;
  if(get_unsteady_evaluation_type(time_control_data) == TimeControlData::UnsteadyEvalType::Timestep)
  {
    if((time_step_number - 1) % time_control_data.trigger_every_time_steps == 0)
    {
      evaluate = true;
      ++counter;
    }
  }
  else if(get_unsteady_evaluation_type(time_control_data) ==
          TimeControlData::UnsteadyEvalType::Interval)
  {
    // In case of a restart, time might be significantly larger than start_time in the first time
    // step of the restarted simulation. Hence, we have to reconstruct the counter at the end of the
    // previous simulation, in order to obtain the correct behavior at the beginning of the
    // restarted simulation.
    if(reset_counter)
    {
      counter += static_cast<unsigned int>((time - time_control_data.start_time + EPSILON) /
                                           time_control_data.trigger_interval);
      reset_counter = false;
    }

    if((time >
        (time_control_data.start_time + counter * time_control_data.trigger_interval - EPSILON)))
    {
      evaluate = true;
      ++counter;
    }
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Not implemented for given TimeControlData::UnsteadyEvalType"));
  }

  return evaluate;
}

unsigned int
TimeControl::get_counter() const
{
  // the counter is incremented during needs_evaluation(), thus counter is ahead by 1
  AssertThrow(counter > 0, dealii::ExcMessage("Counter not in a valid state."));
  return counter - 1;
}

bool
TimeControl::reached_end_time() const
{
  return end_time_reached;
}

bool
TimeControl::get_epsilon() const
{
  return EPSILON;
}


} // namespace ExaDG
