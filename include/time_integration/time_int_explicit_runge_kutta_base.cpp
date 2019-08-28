/*
 * time_int_explicit_runge_kutta_base.cpp
 *
 *  Created on: Nov 19, 2018
 *      Author: fehn
 */

#include "time_int_explicit_runge_kutta_base.h"

template<typename Number>
TimeIntExplRKBase<Number>::TimeIntExplRKBase(double const &      start_time_,
                                             double const &      end_time_,
                                             unsigned int const  max_number_of_time_steps_,
                                             RestartData const & restart_data_,
                                             bool const          adaptive_time_stepping_)
  : TimeIntBase(start_time_, end_time_, max_number_of_time_steps_, restart_data_),
    time_step(1.0),
    adaptive_time_stepping(adaptive_time_stepping_)
{
}

template<typename Number>
void
TimeIntExplRKBase<Number>::reset_time(double const & current_time)
{
  if(current_time <= start_time + eps)
    time = current_time;
  else
    AssertThrow(false, ExcMessage("The variable time may not be overwritten via public access."));
}

template<typename Number>
double
TimeIntExplRKBase<Number>::get_time_step_size() const
{
  return time_step;
}

template<typename Number>
void
TimeIntExplRKBase<Number>::set_time_step_size(double const & time_step_size)
{
  time_step = time_step_size;
}

template<typename Number>
void
TimeIntExplRKBase<Number>::setup(bool const do_restart)
{
  pcout << std::endl << "Setup time integrator ..." << std::endl;

  // initialize time integrator
  initialize_time_integrator();

  // initialize global solution vectors (allocation)
  initialize_vectors();

  if(do_restart)
  {
    // The solution vectors and the current time and the time step size have to be read from restart
    // files.
    read_restart();
  }
  else
  {
    // initializes the solution by interpolation of analytical solution
    initialize_solution();

    // calculate time step size
    calculate_time_step_size();
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<typename Number>
void
TimeIntExplRKBase<Number>::do_timestep(bool const do_write_output)
{
  output_solver_info_header();

  solve_timestep();

  output_remaining_time(do_write_output);

  prepare_vectors_for_next_timestep();

  this->time += time_step;
  ++this->time_step_number;

  if(adaptive_time_stepping == true)
  {
    double const dt = recalculate_time_step_size();
    this->set_time_step_size(dt);
  }

  if(this->restart_data.write_restart == true)
  {
    this->write_restart();
  }
}

template<typename Number>
void
TimeIntExplRKBase<Number>::prepare_vectors_for_next_timestep()
{
  // solution at t_n+1 -> solution at t_n
  solution_n.swap(solution_np);
}

template<typename Number>
void
TimeIntExplRKBase<Number>::do_write_restart(std::string const & filename) const
{
  std::ostringstream oss;

  boost::archive::binary_oarchive oa(oss);

  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // 1. ranks
  oa & n_ranks;

  // 2. time
  oa & time;

  // 3. time step size
  oa & time_step;

  // 4. solution vectors
  oa << solution_n;

  write_restart_file(oss, filename);
}

template<typename Number>
void
TimeIntExplRKBase<Number>::do_read_restart(std::ifstream & in)
{
  boost::archive::binary_iarchive ia(in);

  // Note that the operations done here must be in sync with the output.

  // 1. ranks
  unsigned int n_old_ranks = 1;
  ia &         n_old_ranks;

  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  AssertThrow(n_old_ranks == n_ranks,
              ExcMessage("Tried to restart with " + Utilities::to_string(n_ranks) +
                         " processes, "
                         "but restart was written on " +
                         Utilities::to_string(n_old_ranks) + " processes."));

  // 2. time
  ia & time;

  // Note that start_time has to be set to the new start_time (since param.start_time might still be
  // the original start time).
  this->start_time = time;

  // 3. time step size
  ia & time_step;

  // 4. solution vectors
  ia >> solution_n;
}

// instantiations
template class TimeIntExplRKBase<float>;
template class TimeIntExplRKBase<double>;
