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

#include <exadg/time_integration/time_int_multistep_base.h>

namespace ExaDG
{
TimeIntMultistepBase::TimeIntMultistepBase(double const        start_time_,
                                           double const        end_time_,
                                           unsigned int const  max_number_of_time_steps_,
                                           unsigned int const  order_,
                                           bool const          start_with_low_order_,
                                           bool const          adaptive_time_stepping_,
                                           RestartData const & restart_data_,
                                           MPI_Comm const &    mpi_comm_,
                                           bool const          is_test_)
  : TimeIntBase(start_time_,
                end_time_,
                max_number_of_time_steps_,
                restart_data_,
                mpi_comm_,
                is_test_),
    order(order_),
    start_with_low_order(start_with_low_order_),
    adaptive_time_stepping(adaptive_time_stepping_),
    time_steps(order_, -1.0)
{
}

void
TimeIntMultistepBase::setup(bool const do_restart)
{
  this->pcout << std::endl << "Setup multistep time integrator ..." << std::endl << std::flush;

  // allocate global solution vectors
  allocate_vectors();

  // initializes the solution and the time step size
  initialize_vectors_and_time_step_size(do_restart);

  // this is where the setup of derived classes is performed
  setup_derived();

  this->pcout << std::endl << "... done!" << std::endl;
}

void
TimeIntMultistepBase::initialize_vectors_and_time_step_size(bool do_restart)
{
  if(do_restart)
  {
    if(order > 1 && start_with_low_order == true)
    {
      this->pcout << "WARNING A higher-order time integration scheme is used, " << std::endl
                  << "WARNING but the simulation is started with first-order after the restart."
                  << std::endl
                  << "WARNING Make sure that the parameter start_with_low_order is set correctly."
                  << std::endl;
    }

    // The solution vectors and the current time, the time step size, etc. have to be read from
    // restart files.
    read_restart();
  }
  else
  {
    // The time step size might depend on the current solution.
    initialize_current_solution();

    // The time step size has to be computed before the solution can be initialized at times
    // t - dt[1], t - dt[1] - dt[2], etc.
    time_steps[0] = calculate_time_step_size();

    // initialize time_steps array as a preparation for initialize_former_solutions()
    for(unsigned int i = 1; i < order; ++i)
      time_steps[i] = time_steps[0];

    // Finally, set vectors at former times needed for the multistep method. This is only necessary
    // if the time integrator starts with a high-order scheme in the first time step.
    if(start_with_low_order == false)
      initialize_former_multistep_dof_vectors();
  }
}

void
TimeIntMultistepBase::timeloop_steady_problem()
{
  this->global_timer.restart();

  postprocessing_steady_problem();

  solve_steady_problem();

  postprocessing_steady_problem();
}

void
TimeIntMultistepBase::ale_update()
{
  AssertThrow(false, dealii::ExcMessage("ALE update not implemented."));
}

double
TimeIntMultistepBase::get_previous_time(int const i /* t_{n-i} */) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  AssertThrow(i >= 0, dealii::ExcMessage("Invalid parameter."));

  double t = this->time;

  for(int k = 1; k <= i; ++k)
    t -= this->time_steps[k];

  return t;
}

double
TimeIntMultistepBase::get_time_step_size() const
{
  return get_time_step_size(0);
}

double
TimeIntMultistepBase::get_time_step_size(int const i /* dt[i] */) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}      t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  AssertThrow(i >= 0 && i <= int(order) - 1, dealii::ExcMessage("Invalid access."));

  AssertThrow(time_steps[i] > 0.0, dealii::ExcMessage("Invalid or uninitialized time step size."));

  return time_steps[i];
}

std::vector<double>
TimeIntMultistepBase::get_time_step_vector() const
{
  return time_steps;
}

void
TimeIntMultistepBase::push_back_time_step_sizes()
{
  /*
   * push back time steps
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   *
   *                    dt[1]  <- dt[0] <- new_dt
   *
   */
  for(unsigned int i = order - 1; i > 0; --i)
    time_steps[i] = time_steps[i - 1];
}

void
TimeIntMultistepBase::set_current_time_step_size(double const & time_step_size)
{
  // constant time step sizes: allow setting of time step size only in the first
  // time step or after a restart (where time_step_number is also 1), e.g., to continue
  // will smaller time step sizes
  if(adaptive_time_stepping == false)
  {
    AssertThrow(time_step_number == 1,
                dealii::ExcMessage("For time integration with constant time step sizes this "
                                   "function can only be called in the very first time step."));
  }

  time_steps[0] = time_step_size;
}

void
TimeIntMultistepBase::do_timestep_pre_solve(bool const print_header)
{
  if(this->print_solver_info() && print_header)
    this->output_solver_info_header();

  update_time_integrator_constants();
}

void
TimeIntMultistepBase::do_timestep_post_solve()
{
  prepare_vectors_for_next_timestep();

  time += time_steps[0];
  ++time_step_number;

  if(adaptive_time_stepping == true)
  {
    push_back_time_step_sizes();
    time_steps[0] = recalculate_time_step_size();
  }

  if(restart_data.write_restart == true)
  {
    write_restart();
  }

  if(this->print_solver_info())
  {
    output_remaining_time();
  }
}

void
TimeIntMultistepBase::do_read_restart(std::ifstream & in)
{
  boost::archive::binary_iarchive ia(in);
  read_restart_preamble(ia);
  read_restart_vectors(ia);

  // In order to change the CFL number (or the time step calculation criterion in general),
  // start_with_low_order = true has to be used. Otherwise, the old solutions would not fit the
  // time step increments.
  if(start_with_low_order == true)
    time_steps[0] = calculate_time_step_size();
}

void
TimeIntMultistepBase::read_restart_preamble(boost::archive::binary_iarchive & ia)
{
  // Note that the operations done here must be in sync with the output.

  // 1. ranks
  unsigned int n_old_ranks = 1;
  ia &         n_old_ranks;

  unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
  AssertThrow(n_old_ranks == n_ranks,
              dealii::ExcMessage("Tried to restart with " + dealii::Utilities::to_string(n_ranks) +
                                 " processes, "
                                 "but restart was written on " +
                                 dealii::Utilities::to_string(n_old_ranks) + " processes."));

  // 2. time
  ia & time;

  // Note that start_time has to be set to the new start_time (since param.start_time might still be
  // the original start time).
  this->start_time = time;

  // 3. order
  unsigned int old_order = 1;
  ia &         old_order;

  AssertThrow(old_order == order, dealii::ExcMessage("Order of time integrator may not change."));

  // 4. time step sizes
  for(unsigned int i = 0; i < order; i++)
    ia & time_steps[i];
}

void
TimeIntMultistepBase::do_write_restart(std::string const & filename) const
{
  std::ostringstream oss;

  boost::archive::binary_oarchive oa(oss);

  write_restart_preamble(oa);
  write_restart_vectors(oa);
  write_restart_file(oss, filename);
}

void
TimeIntMultistepBase::write_restart_preamble(boost::archive::binary_oarchive & oa) const
{
  unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  // 1. ranks
  oa & n_ranks;

  // 2. time
  oa & time;

  // 3. order
  oa & order;

  // 4. time step sizes
  for(unsigned int i = 0; i < order; i++)
    oa & time_steps[i];
}

void
TimeIntMultistepBase::postprocessing_steady_problem() const
{
  AssertThrow(false, dealii::ExcMessage("This function has to be implemented by derived classes."));
}

void
TimeIntMultistepBase::solve_steady_problem()
{
  AssertThrow(false, dealii::ExcMessage("This function has to be implemented by derived classes."));
}

} // namespace ExaDG
