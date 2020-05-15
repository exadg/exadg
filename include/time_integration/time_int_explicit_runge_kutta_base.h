/*
 * time_int_explicit_runge_kutta_base.h
 *
 *  Created on: Nov 19, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_BASE_H_
#define INCLUDE_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_BASE_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "time_integration/time_int_base.h"

using namespace dealii;

template<typename Number>
class TimeIntExplRKBase : public TimeIntBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  TimeIntExplRKBase(double const &      start_time_,
                    double const &      end_time_,
                    unsigned int const  max_number_of_time_steps_,
                    RestartData const & restart_data_,
                    bool const          adaptive_time_stepping_,
                    MPI_Comm const &    mpi_comm_);

  void
  setup(bool const do_restart = false);

  double
  get_time_step_size() const;

  void
  set_current_time_step_size(double const & time_step_size);

protected:
  // solution vectors
  VectorType solution_n, solution_np;

  // time step size
  double time_step;

  // use adaptive time stepping?
  bool const adaptive_time_stepping;

private:
  void
  do_timestep_pre_solve(bool const print_header);

  void
  do_timestep_post_solve();

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
  do_write_restart(std::string const & filename) const override;

  void
  do_read_restart(std::ifstream & in) override;
};



#endif /* INCLUDE_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_BASE_H_ */
