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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/bdf_time_integration.h>
#include <exadg/time_integration/extrapolation_scheme.h>
#include <exadg/time_integration/time_int_base.h>

namespace ExaDG
{
template<typename Number>
class TimeIntBDFBase : public TimeIntBase
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /*
   * Constructor.
   */
  TimeIntBDFBase(double const        start_time_,
                 double const        end_time_,
                 unsigned int const  max_number_of_time_steps_,
                 unsigned const      order_,
                 bool const          start_with_low_order_,
                 bool const          adaptive_time_stepping_,
                 RestartData const & restart_data_,
                 MPI_Comm const &    mpi_comm_,
                 bool const          is_test_);

  /*
   * Destructor.
   */
  virtual ~TimeIntBDFBase()
  {
  }

  /*
   * Setup function where allocations/initializations are done. Calls another function
   * setup_derived() in which the setup of derived classes can be performed.
   */
  void
  setup(bool const do_restart) final;

  /*
   * Pseudo-time-stepping for steady-state problems.
   */
  void
  timeloop_steady_problem();

  /*
   * Setters and getters.
   */
  double
  get_scaling_factor_time_derivative_term() const;

  /*
   * Get the time step size.
   */
  double
  get_time_step_size() const final;

  double
  get_time_step_size(int const index) const;

  /*
   * Set the time step size. Note that the push-back of time step sizes in case of adaptive time
   * stepping may not be done here because calling this public function several times would falsify
   * the results. Hence, set_time_step_size() is only allowed to overwrite the current time step
   * size.
   */
  void
  set_current_time_step_size(double const & time_step_size) final;

  /*
   * Get time at the end of the current time step.
   */
  double
  get_previous_time(int const i /* t_{n-i} */) const;

protected:
  /*
   * Do one time step including different updates before and after the actual solution of the
   * current time step.
   */
  void
  do_timestep_pre_solve(bool const print_header) final;

  void
  do_timestep_post_solve() final;

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
   * Calculate time step size.
   */
  virtual double
  calculate_time_step_size() = 0;

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
   * Vector with time step sizes.
   */
  std::vector<double> time_steps;

private:
  /*
   * Allocate solution vectors (has to be implemented by derived classes).
   */
  virtual void
  allocate_vectors() = 0;

  /*
   * Initializes the solution vectors by prescribing initial conditions or reading data from
   * restart files and initializes the time step size.
   */
  virtual void
  initialize_solution_and_time_step_size(bool do_restart);

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
   * Solve for a steady-state solution using pseudo-time-stepping.
   */
  virtual void
  solve_steady_problem();

  /*
   * Postprocessing of solution.
   */
  virtual void
  postprocessing_steady_problem() const;

  /*
   * Restart: read solution vectors (has to be implemented in derived classes).
   */
  void
  do_read_restart(std::ifstream & in) final;

  void
  read_restart_preamble(boost::archive::binary_iarchive & ia);

  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia) = 0;

  /*
   * Write solution vectors to files so that the simulation can be restart from an intermediate
   * state.
   */
  void
  do_write_restart(std::string const & filename) const final;

  void
  write_restart_preamble(boost::archive::binary_oarchive & oa) const;

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const = 0;

  /*
   * Recalculate the time step size after each time step in case of adaptive time stepping.
   */
  virtual double
  recalculate_time_step_size() const = 0;

  /*
   * returns whether solver info has to be written in the current time step.
   */
  virtual bool
  print_solver_info() const = 0;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_ */
