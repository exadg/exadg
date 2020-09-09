/*
 * restart_data.h
 *
 *  Created on: Nov 13, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_RESTART_DATA_H_
#define INCLUDE_FUNCTIONALITIES_RESTART_DATA_H_

// C/C++
#include <limits>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
using namespace dealii;

struct RestartData
{
  RestartData()
    : write_restart(false),
      interval_time(std::numeric_limits<double>::max()),
      interval_wall_time(std::numeric_limits<double>::max()),
      interval_time_steps(std::numeric_limits<unsigned int>::max()),
      filename("restart"),
      counter(1)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    pcout << "  Restart:" << std::endl;
    print_parameter(pcout, "Write restart", write_restart);

    if(write_restart == true)
    {
      print_parameter(pcout, "Interval physical time", interval_time);
      print_parameter(pcout, "Interval wall time", interval_wall_time);
      print_parameter(pcout, "Interval time steps", interval_time_steps);
      print_parameter(pcout, "Filename", filename);
    }
  }

  bool
  do_restart(double const       wall_time,
             double const       time,
             unsigned int const time_step_number,
             bool const         reset_counter) const
  {
    // After a restart, the counter is reset to 1, but time = current_time - start time != 0 after a
    // restart. Hence, we have to explicitly reset the counter in that case. There is nothing to do
    // if the restart is controlled by the wall time or the time_step_number because these
    // variables are reinitialized after a restart anyway.
    if(reset_counter)
      counter += int((time + 1.e-10) / interval_time);

    bool do_restart = wall_time > interval_wall_time * counter || time > interval_time * counter ||
                      time_step_number > interval_time_steps * counter;

    if(do_restart)
      ++counter;

    return do_restart;
  }

  bool write_restart;

  // physical time
  double interval_time;

  // wall time in seconds (= hours * 3600)
  double interval_wall_time;

  // number of time steps after which to write restart
  unsigned int interval_time_steps;

  // filename for restart files
  std::string filename;

  // counter needed do decide when to write restart
  mutable unsigned int counter;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_RESTART_DATA_H_ */
