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

#ifndef EXADG_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_BASE_H_
#define EXADG_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/time_int_base.h>

namespace ExaDG
{
template<typename Number>
class TimeIntExplRKBase : public TimeIntBase
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  TimeIntExplRKBase(double const &      start_time_,
                    double const &      end_time_,
                    unsigned int const  max_number_of_time_steps_,
                    RestartData const & restart_data_,
                    bool const          adaptive_time_stepping_,
                    MPI_Comm const &    mpi_comm_,
                    bool const          is_test_);

  void
  setup(bool const do_restart) final;

  double
  get_time_step_size() const final;

  void
  set_current_time_step_size(double const & time_step_size) final;

protected:
  // solution vectors
  VectorType solution_n, solution_np;

  // time step size
  double time_step;

  // use adaptive time stepping?
  bool const adaptive_time_stepping;

private:
  void
  do_timestep_pre_solve(bool const print_header) final;

  void
  do_timestep_post_solve() final;

  void
  prepare_vectors_for_next_timestep();

  virtual void
  initialize_time_integrator() = 0;

  virtual void
  initialize_vectors() = 0;

  virtual void
  initialize_solution() = 0;

  virtual void
  calculate_time_step_size() = 0;

  virtual double
  recalculate_time_step_size() const = 0;

  /*
   * returns whether solver info has to be written in the current time step.
   */
  virtual bool
  print_solver_info() const = 0;

  void
  do_write_restart(std::string const & filename) const final;

  void
  do_read_restart(std::ifstream & in) final;

  virtual void
  read_restart_vectors(std::vector<VectorType *> const & vectors);

  virtual void
  write_restart_vectors(std::vector<VectorType const *> const & vectors) const;
};

} // namespace ExaDG

#endif /* EXADG_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_BASE_H_ */
