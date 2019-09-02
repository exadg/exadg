/*
 * time_int_base.cpp
 *
 *  Created on: Nov 16, 2018
 *      Author: fehn
 */

#include "time_int_base.h"

TimeIntBase::TimeIntBase(double const &      start_time_,
                         double const &      end_time_,
                         unsigned int const  max_number_of_time_steps_,
                         RestartData const & restart_data_)
  : start_time(start_time_),
    end_time(end_time_),
    time(start_time_),
    eps(1.e-10),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_step_number(1),
    max_number_of_time_steps(max_number_of_time_steps_),
    restart_data(restart_data_)
{
}

void
TimeIntBase::timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  while(time < (end_time - eps) && time_step_number <= max_number_of_time_steps)
  {
    do_timestep();

    postprocessing();
  }

  pcout << std::endl << "... finished time loop!" << std::endl;
}

bool
TimeIntBase::advance_one_timestep(bool write_final_output)
{
  bool started = time > (start_time - eps);

  // If the time integrator has not yet started, simply increment physical time without solving the
  // current time step.
  if(!started)
  {
    time += get_time_step_size();
  }

  if(started && time_step_number == 1)
  {
    pcout << std::endl << "Starting time loop ..." << std::endl;

    global_timer.restart();

    postprocessing();
  }

  // check if we have reached the end of the time loop
  bool finished = !(time < (end_time - eps) && time_step_number <= max_number_of_time_steps);

  // advance one time step and perform postprocessing
  if(started && !finished)
  {
    do_timestep();

    postprocessing();
  }

  // for the statistics
  if(finished && write_final_output)
  {
    pcout << std::endl << "... done!" << std::endl;
  }

  return finished;
}

double
TimeIntBase::get_time() const
{
  return time;
}

double
TimeIntBase::get_number_of_time_steps() const
{
  return this->get_time_step_number() - 1;
}

unsigned int
TimeIntBase::get_time_step_number() const
{
  return time_step_number;
}

void
TimeIntBase::write_restart() const
{
  double const wall_time = global_timer.wall_time();

  if(restart_data.do_restart(wall_time, time - start_time, time_step_number, time_step_number == 2))
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Writing restart file at time t = " << this->get_time() << ":" << std::endl;

    std::string const filename = restart_filename(restart_data.filename);

    rename_restart_files(filename);

    do_write_restart(restart_filename(restart_data.filename));

    pcout << std::endl
          << " ... done!" << std::endl
          << "______________________________________________________________________" << std::endl;
  }
}

void
TimeIntBase::read_restart()
{
  pcout << std::endl
        << "______________________________________________________________________" << std::endl
        << std::endl
        << " Reading restart file:" << std::endl;

  std::string   filename = restart_filename(restart_data.filename);
  std::ifstream in(filename);
  AssertThrow(in, ExcMessage("File " + filename + " does not exist."));

  do_read_restart(in);

  pcout << std::endl
        << " ... done!" << std::endl
        << "______________________________________________________________________" << std::endl
        << std::endl;
}

void
TimeIntBase::output_solver_info_header() const
{
  if(print_solver_info())
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << time_step_number
          << "t_n = " << std::scientific << std::setprecision(4) << time
          << " -> t_n+1 = " << time + get_time_step_size() << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl;
  }
}

/*
 *  This function estimates the remaining wall time based on the overall time interval to be
 *  simulated and the measured wall time already needed to simulate from the start time until the
 *  current time.
 */
void
TimeIntBase::output_remaining_time(bool const do_write_output) const
{
  if(print_solver_info() && do_write_output)
  {
    if(time > start_time)
    {
      double const remaining_time =
        global_timer.wall_time() * (end_time - time) / (time - start_time);

      int const hours   = int(remaining_time / 3600.0);
      int const minutes = int((remaining_time - hours * 3600.0) / 60.0);
      int const seconds = int((remaining_time - hours * 3600.0 - minutes * 60.0));

      pcout << std::endl
            << "Estimated time until completion is " << hours << " h " << minutes << " min "
            << seconds << " s." << std::endl;
    }
  }
}
