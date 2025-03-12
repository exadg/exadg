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

#include <exadg/time_integration/time_int_base.h>
#include <iostream>

namespace ExaDG
{
TimeIntBase::TimeIntBase(double const &      start_time_,
                         double const &      end_time_,
                         unsigned int const  max_number_of_time_steps_,
                         RestartData const & restart_data_,
                         MPI_Comm const &    mpi_comm_,
                         bool const          is_test_)
  : start_time(start_time_),
    end_time(end_time_),
    time(start_time_),
    eps(1.e-10),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    time_step_number(1),
    max_number_of_time_steps(max_number_of_time_steps_),
    restart_data(restart_data_),
    mpi_comm(mpi_comm_),
    timer_tree(new TimerTree()),
    is_test(is_test_)
{
}

bool
TimeIntBase::started() const
{
  return time > (start_time - eps);
}

bool
TimeIntBase::finished() const
{
  return not(time < (end_time - eps) and time_step_number <= max_number_of_time_steps);
}

void
TimeIntBase::timeloop()
{
  while(not finished())
  {
    advance_one_timestep();
  }
}

void
TimeIntBase::advance_one_timestep()
{
  advance_one_timestep_pre_solve(true);

  advance_one_timestep_solve();

  advance_one_timestep_post_solve();
}

void
TimeIntBase::advance_one_timestep_pre_solve(bool const print_header)
{
  dealii::Timer timer;
  timer.restart();

  if(started() and not finished())
  {
    if(time_step_number == 1)
    {
      global_timer.restart();

      pcout << std::endl << "Starting time loop ..." << std::endl;

      postprocessing();
    }

    do_timestep_pre_solve(print_header);
  }

  timer_tree->insert({"Timeloop"}, timer.wall_time());
}

void
TimeIntBase::advance_one_timestep_solve()
{
  dealii::Timer timer;
  timer.restart();

  if(started() and not finished())
  {
    do_timestep_solve();
  }

  timer_tree->insert({"Timeloop"}, timer.wall_time());
}

void
TimeIntBase::advance_one_timestep_post_solve()
{
  dealii::Timer timer;
  timer.restart();

  if(started() and not finished())
  {
    do_timestep_post_solve();

    postprocessing();
  }
  else
  {
    // If the time integrator is not "active", simply increment time.
    time += get_time_step_size();
  }

  timer_tree->insert({"Timeloop"}, timer.wall_time());
}

void
TimeIntBase::reset_time(double const & current_time)
{
  // Only allow overwriting the time to a value smaller than start_time
  // (which is needed when coupling different solvers, different domains, etc.).
  if(current_time <= start_time + eps)
    time = current_time;
  else
    AssertThrow(false,
                dealii::ExcMessage("The variable time may not be overwritten via public access."));
}

void
TimeIntBase::prepare_coarsening_and_refinement()
{
  AssertThrow(false, dealii::ExcMessage("Overwrite in derived class to enable adaptivity."));
}

void
TimeIntBase::interpolate_after_coarsening_and_refinement()
{
  AssertThrow(false, dealii::ExcMessage("Overwrite in derived class to enable adaptivity."));
}

double
TimeIntBase::get_time() const
{
  return time;
}

double
TimeIntBase::get_next_time() const
{
  return this->get_time() + this->get_time_step_size();
}

unsigned int
TimeIntBase::get_number_of_time_steps() const
{
  return this->get_time_step_number() - 1;
}

std::shared_ptr<TimerTree>
TimeIntBase::get_timings() const
{
  return timer_tree;
}

void
TimeIntBase::do_timestep()
{
  do_timestep_pre_solve(true);

  do_timestep_solve();

  do_timestep_post_solve();
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
          << print_horizontal_line() << std::endl
          << std::endl
          << " Writing restart file at time t = " << this->get_time() << ":" << std::endl;

    std::string const filename = restart_filename(restart_data.filename, mpi_comm);

    rename_restart_files(filename);

    do_write_restart(restart_filename(restart_data.filename, mpi_comm));

    pcout << std::endl << " ... done!" << std::endl << print_horizontal_line() << std::endl;
  }
}

void
TimeIntBase::read_restart()
{
  pcout << std::endl
        << print_horizontal_line() << std::endl
        << std::endl
        << " Reading restart file:" << std::endl;

  std::string   filename = restart_filename(restart_data.filename, mpi_comm);
  std::ifstream in(filename);
  AssertThrow(in, dealii::ExcMessage("File " + filename + " does not exist."));

  do_read_restart(in);

  pcout << std::endl
        << " ... done!" << std::endl
        << print_horizontal_line() << std::endl
        << std::endl;
}

void
TimeIntBase::output_solver_info_header() const
{
  pcout << std::endl
        << print_horizontal_line() << std::endl
        << std::endl
        << " Time step number = " << std::left << std::setw(8) << time_step_number
        << "t = " << std::scientific << std::setprecision(5) << time
        << " -> t + dt = " << time + get_time_step_size() << std::endl
        << print_horizontal_line() << std::endl;
}

/*
 *  This function estimates the remaining wall time based on the overall time interval to be
 *  simulated and the measured wall time already needed to simulate from the start time until the
 *  current time.
 */
void
TimeIntBase::output_remaining_time() const
{
  if(not(this->is_test))
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

} // namespace ExaDG
