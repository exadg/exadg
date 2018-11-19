/*
 * time_int_bdf_base.h
 *
 *  Created on: Nov 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_
#define INCLUDE_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include "time_integration/bdf_time_integration.h"
#include "time_integration/extrapolation_scheme.h"

#include "functionalities/restart_data.h"
#include "time_integration/restart.h"

using namespace dealii;

class TimeIntBDFBase
{
public:
  /*
   * Constructor.
   */
  TimeIntBDFBase(double const        start_time_,
                 double const        end_time_,
                 unsigned int const  max_number_of_time_steps_,
                 double const        order_,
                 bool const          start_with_low_order_,
                 bool const          adaptive_time_stepping_,
                 RestartData const & restart_data_);

  /*
   * Destructor.
   */
  virtual ~TimeIntBDFBase()
  {
  }

  /*
   * Perform whole time loop from start time to end time.
   */
  void
  timeloop();

  /*
   * Pseudo-time-stepping for steady-state problems.
   */
  void
  timeloop_steady_problem();

  /*
   * Perform only one time step (which is used when coupling different solvers, equations, etc.).
   */
  bool
  advance_one_timestep(bool write_final_output);

  /*
   * Setters and getters.
   */
  double
  get_scaling_factor_time_derivative_term() const;

  /*
   * Reset the current time.
   */
  void
  reset_time(double const & current_time);

  /*
   * Get the current time t_{n}.
   */
  double
  get_time() const;

  /*
   * Get the time step size.
   */
  double
  get_time_step_size(int const index = 0) const;

  /*
   * Set the time step size. Note that the push-back of time step sizes in case of adaptive time
   * stepping may not be done here because calling this public function several times would falsify
   * the results. Hence, set_time_step_size() is only allowed to overwrite the current time step
   * size.
   */
  void
  set_time_step_size(double const & time_step);

  /*
   * Setup function where allocations/initializations are done. Calls another function
   * setup_derived() in which the setup of derived classes can be performed.
   */
  void
  setup(bool const do_restart);

protected:
  /*
   * Get time at the end of the current time step t_{n+1}.
   */
  double
  get_next_time() const;

  /*
   * Get time at the end of the current time step.
   */
  double
  get_previous_time(int const i /* t_{n-i} */) const;

  /*
   * Get the current time step number.
   */
  unsigned int
  get_time_step_number() const;

  /*
   * Do one time step including different updates before and after the actual solution of the
   * current time step.
   */
  void
  do_timestep();

  /*
   * Update the time integrator constants.
   */
  virtual void
  update_time_integrator_constants();

  /*
   * Get reference to vector with time step sizes
   */
  std::vector<double>
  get_time_step_vector() const;

  /*
   * Update of time step sizes in case of variable time steps.
   */
  void
  push_back_time_step_sizes();

  /*
   * This function implements the OIF sub-stepping algorithm.
   */
  void
  calculate_sum_alphai_ui_oif_substepping(double const cfl, double const cfl_oif);

  /*
   * Read all relevant data from restart files to start the time integrator.
   */
  virtual void
  read_restart();

  /*
   * Calculate time step size.
   */
  virtual void
  calculate_time_step_size() = 0;

  /*
   * Start and end times.
   */
  double const start_time, end_time;

  /*
   * Maximum number of time steps.
   */
  unsigned int const max_number_of_time_steps;

  /*
   * Order of time integration scheme.
   */
  unsigned int const order;

  /*
   * Time integration constants. The extrapolation scheme is not necessarily used for a BDF time
   * integration scheme with fully implicit time stepping, implying a violation of the Liskov
   * substitution principle (OO software design principle). However, it does not appear to be
   * reasonable to complicate the inheritance due to this fact.
   */
  BDFTimeIntegratorConstants bdf;
  ExtrapolationConstants     extra;

  /*
   * Start with low order (1st order) time integration scheme in first time step.
   */
  bool const start_with_low_order;

  /*
   * Use adaptive time stepping?
   */
  bool const adaptive_time_stepping;

  /*
   * Physical time.
   */
  double time;

  /*
   * Vector with time step sizes.
   */
  std::vector<double> time_steps;

  /*
   * Computation time (wall clock time).
   */
  Timer  global_timer;
  double total_time;

  /*
   * A small number which is much smaller than the time step size.
   */
  double const eps;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

private:
  /*
   * Allocate solution vectors (has to be implemented by derived classes).
   */
  virtual void
  allocate_vectors() = 0;

  /*
   * Initializes everything related to OIF sub-stepping.
   */
  virtual void
  initialize_oif() = 0;

  /*
   * Initializes the solution vectors by prescribing initial conditions are reading data from
   * restart files and calculates the time step size.
   */
  virtual void
  initialize_solution_and_calculate_timestep(bool do_restart);

  /*
   * Initializes the solution vectors at time t
   */
  virtual void
  initialize_current_solution() = 0;

  /*
   * Initializes the solution vectors at time t - dt[1], t - dt[1] - dt[2], etc.
   */
  virtual void
  initialize_former_solutions() = 0;

  /*
   * Setup of derived classes.
   */
  virtual void
  setup_derived() = 0;

  /*
   * This function prepares the solution vectors for the next time step, e.g., by switching pointers
   * to the solution vectors (called push back here).
   */
  virtual void
  prepare_vectors_for_next_timestep() = 0;

  /*
   * Solve the current time step.
   */
  virtual void
  solve_timestep() = 0;

  /*
   * Solve for a steady-state solution using pseudo-time-stepping.
   */
  virtual void
  solve_steady_problem();


  /*
   * Output solver information before solving the time step
   */
  virtual void
  output_solver_info_header() const = 0;

  /*
   * Output estimated computation time until completion of the simulation.
   */
  virtual void
  output_remaining_time() const = 0;

  /*
   * Postprocessing of solution.
   */
  virtual void
  postprocessing() const = 0;

  virtual void
  postprocessing_steady_problem() const;

  /*
   * Analysis of computation times called after having performed the time loop.
   */
  virtual void
  analyze_computing_times() const = 0;

  /*
   * Restart: read solution vectors (has to be implemented in derived classes).
   */
  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia) = 0;

  /*
   * Restart: write solution vectors (has to be implemented in derived classes).
   */
  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const = 0;

  /*
   * Write solution vectors to files so that the simulation can be restart from an intermediate
   * state.
   */
  virtual void
  write_restart() const;

  /*
   * Recalculate the time step size after each time step in case of adaptive time stepping.
   */
  virtual double
  recalculate_time_step() = 0;

  /*
   * Initializes the solution for OIF sub-stepping at time t_{n-i}.
   */
  virtual void
  initialize_solution_oif_substepping(unsigned int i);

  /*
   * Adds result of OIF sub-stepping for outer loop index i to sum_alphai_ui.
   */
  virtual void
  update_sum_alphai_ui_oif_substepping(unsigned int i);

  /*
   * Perform one timestep for OIF sub-stepping and update the solution vectors (switch pointers).
   */
  virtual void
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size);

private:
  /*
   * The number of the current time step starting with time_step_number = 1.
   */
  unsigned int time_step_number;

  /*
   * Restart.
   */
  RestartData const restart_data;
};

#endif /* INCLUDE_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_ */
