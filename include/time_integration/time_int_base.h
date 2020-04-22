/*
 * time_int_base.h
 *
 *  Created on: Nov 16, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_TIME_INT_BASE_H_
#define INCLUDE_TIME_INTEGRATION_TIME_INT_BASE_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <fstream>
#include <sstream>

#include "functionalities/restart_data.h"
#include "time_integration/restart.h"

using namespace dealii;

class TimeIntBase
{
public:
  TimeIntBase(double const &      start_time_,
              double const &      end_time_,
              unsigned int const  max_number_of_time_steps_,
              RestartData const & restart_data_,
              MPI_Comm const &    mpi_comm_);

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
  advance_one_timestep_pre_solve();

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
   * Get number of computed time steps
   */
  double
  get_number_of_time_steps() const;

  /*
   * Do one time step including different updates before and after the actual solution of the
   * current time step.
   */
  void
  do_timestep(bool const do_write_output = true);

  /*
   * e.g., update of time integrator constants
   */
  virtual void
  do_timestep_pre_solve() = 0;

  virtual void
  solve_timestep() = 0;

  /*
   * e.g., update of DoF vectors, increment time, adjust time step size, etc.
   */
  virtual void
  do_timestep_post_solve(bool const do_write_output = true) = 0;

  /*
   * Postprocessing of solution.
   */
  virtual void
  postprocessing() const = 0;

protected:
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
   * Computation time (wall clock time).
   */
  Timer global_timer;

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

  MPI_Comm const & mpi_comm;

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



#endif /* INCLUDE_TIME_INTEGRATION_TIME_INT_BASE_H_ */
