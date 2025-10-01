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

#include <exadg/time_integration/restart.h>
#include <exadg/time_integration/time_int_explicit_runge_kutta_base.h>

namespace ExaDG
{
template<typename Number>
TimeIntExplRKBase<Number>::TimeIntExplRKBase(double const &      start_time_,
                                             double const &      end_time_,
                                             unsigned int const  max_number_of_time_steps_,
                                             RestartData const & restart_data_,
                                             bool const          adaptive_time_stepping_,
                                             MPI_Comm const &    mpi_comm_,
                                             bool const          is_test_)
  : TimeIntBase(start_time_,
                end_time_,
                max_number_of_time_steps_,
                restart_data_,
                mpi_comm_,
                is_test_),
    time_step(1.0),
    adaptive_time_stepping(adaptive_time_stepping_)
{
}

template<typename Number>
double
TimeIntExplRKBase<Number>::get_time_step_size() const
{
  return time_step;
}

template<typename Number>
void
TimeIntExplRKBase<Number>::set_current_time_step_size(double const & time_step_size)
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
TimeIntExplRKBase<Number>::do_timestep_pre_solve(bool const print_header)
{
  if(this->print_solver_info() and print_header)
    this->output_solver_info_header();
}

template<typename Number>
void
TimeIntExplRKBase<Number>::do_timestep_post_solve()
{
  prepare_vectors_for_next_timestep();

  this->time += time_step;
  ++this->time_step_number;

  if(adaptive_time_stepping == true)
  {
    this->time_step = recalculate_time_step_size();
  }

  if(this->restart_data.write_restart == true)
  {
    this->write_restart();
  }

  if(this->print_solver_info())
  {
    this->output_remaining_time();
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

  BoostOutputArchiveType oa(oss);

  // 1. time
  oa & time;

  // 2. time step size
  oa & time_step;

  // 3. solution vectors
  std::vector<VectorType const *> vectors{&solution_n};
  this->write_restart_vectors(vectors);

  write_restart_file(oss, filename);
}

template<typename Number>
void
TimeIntExplRKBase<Number>::do_read_restart(std::ifstream & in)
{
  BoostInputArchiveType ia(in);

  // Note that the operations done here must be in sync with the output.

  // 1. time
  ia & time;

  // Note that start_time has to be set to the new start_time (since param.start_time might still be
  // the original start time).
  this->start_time = time;

  // 2. time step size
  ia & time_step;

  // 3. solution vectors
  std::vector<VectorType *> vectors{&solution_n};
  this->read_restart_vectors(vectors);
}

template<typename Number>
void
TimeIntExplRKBase<Number>::read_restart_vectors(std::vector<VectorType *> const & vectors)
{
  (void)vectors;
  AssertThrow(false,
              dealii::ExcMessage("Overwrite this method in the derived "
                                 "class to enable de-/serialization."));
}

template<typename Number>
void
TimeIntExplRKBase<Number>::write_restart_vectors(
  std::vector<VectorType const *> const & vectors) const
{
  (void)vectors;
  AssertThrow(false,
              dealii::ExcMessage("Overwrite this method in the derived "
                                 "class to enable de-/serialization."));
}

// instantiations
template class TimeIntExplRKBase<float>;
template class TimeIntExplRKBase<double>;

} // namespace ExaDG
