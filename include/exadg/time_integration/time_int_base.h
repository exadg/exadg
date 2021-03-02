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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_BASE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_BASE_H_

// C/C++
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/time_integration/restart.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/utilities/timer_tree.h>


namespace ExaDG
{
using namespace dealii;

class TimeIntBase
{
public:
  TimeIntBase(double const &      start_time_,
              double const &      end_time_,
              unsigned int const  max_number_of_time_steps_,
              RestartData const & restart_data_,
              MPI_Comm const &    mpi_comm_,
              bool const          print_wall_times_);

  virtual ~TimeIntBase()
  {
  }

  /*
   * Setup of time integration scheme.
   */
  virtual void
  setup(bool const do_restart) = 0;

  /*
   * Returns true if the start time has been reached.
   */
  bool
  started() const;

  /*
   * returns true if the end of time loop has been reached or the maximum number of time steps
   */
  bool
  finished() const;

  /*
   * Performs the time loop from start_time to end_time.
   */
  void
  timeloop();

  /*
   * Perform only one time step (which is used when coupling different solvers, equations, etc.).
   */
  void
  advance_one_timestep_pre_solve(bool const print_header);

  void
  advance_one_timestep_solve();

  void
  advance_one_timestep_post_solve();

  void
  advance_one_timestep();

  /*
   * Reset the current time.
   */
  void
  reset_time(double const & current_time);

  /*
   * Get the time step size.
   */
  virtual double
  get_time_step_size() const = 0;

  /*
   * Set the time step size.
   */
  virtual void
  set_current_time_step_size(double const & time_step_size) = 0;

  /*
   * Get the current time t_{n}.
   */
  double
  get_time() const;

  /*
   * Get time at the end of the current time step t_{n+1}.
   */
  double
  get_next_time() const;

  /*
   * Get number of computed time steps
   */
  unsigned int
  get_number_of_time_steps() const;

  std::shared_ptr<TimerTree>
  get_timings() const;

protected:
  /*
   * Do one time step including different updates before and after the actual solution of the
   * current time step.
   */
  void
  do_timestep();

  /*
   * e.g., update of time integrator constants
   */
  virtual void
  do_timestep_pre_solve(bool const print_header) = 0;

  virtual void
  solve_timestep() = 0;

  /*
   * e.g., update of DoF vectors, increment time, adjust time step size, etc.
   */
  virtual void
  do_timestep_post_solve() = 0;

  /*
   * Postprocessing of solution.
   */
  virtual void
  postprocessing() const = 0;

  /*
   * Get the current time step number.
   */
  unsigned int
  get_time_step_number() const;

  /*
   * Write solution vectors to files so that the simulation can be restart from an intermediate
   * state.
   */
  void
  write_restart() const;

  /*
   * Read all relevant data from restart files to start the time integrator.
   */
  void
  read_restart();

  /*
   * Output solver information before solving the time step.
   */
  void
  output_solver_info_header() const;


  /*
   * Output estimated computation time until completion of the simulation.
   */
  void
  output_remaining_time() const;

  /*
   * Start and end times.
   */
  double start_time, end_time;

  /*
   * Physical time.
   */
  double time;

  /*
   * A small number which is much smaller than the time step size.
   */
  double const eps;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

  /*
   * The number of the current time step starting with time_step_number = 1.
   */
  unsigned int time_step_number;

  /*
   * Maximum number of time steps.
   */
  unsigned int const max_number_of_time_steps;

  /*
   * Restart.
   */
  RestartData const restart_data;

  /*
   * MPI communicator.
   */
  MPI_Comm const mpi_comm;

  /*
   * Computation time (wall clock time).
   */
  Timer                      global_timer;
  std::shared_ptr<TimerTree> timer_tree;
  bool                       print_wall_times;

private:
  /*
   * Write restart data.
   */
  virtual void
  do_write_restart(std::string const & filename) const = 0;

  /*
   * Read restart data.
   */
  virtual void
  do_read_restart(std::ifstream & in) = 0;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_BASE_H_ */
