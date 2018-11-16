/*
 * time_int_explicit_runge_kutta.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "time_integration/explicit_runge_kutta.h"

using namespace dealii;

namespace ConvDiff
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class Operator;
}

template<typename Number>
class TimeIntExplRK
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  TimeIntExplRK(std::shared_ptr<Operator> operator_in,
                InputParameters const &   param_in,
                unsigned int const        n_refine_time_in);

  void
  setup();

  void
  timeloop();

  bool
  advance_one_timestep(bool write_final_output);

  void
  reset_time(double const & current_time);

  double
  get_time_step_size() const;

  void
  set_time_step_size(double const time_step_size);

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  postprocessing() const;

  void
  output_solver_info_header();

  void
  output_remaining_time();

  void
  do_timestep();

  void
  solve_timestep();

  void
  prepare_vectors_for_next_timestep();

  void
  calculate_timestep();

  void
  initialize_time_integrator();

  void
  analyze_computing_times() const;

  std::shared_ptr<Operator> pde_operator;

  std::shared_ptr<ExplicitTimeIntegrator<Operator, VectorType>> rk_time_integrator;

  InputParameters const & param;

  // timer
  Timer  global_timer;
  double total_time;

  // screen output
  ConditionalOStream pcout;

  // solution vectors
  VectorType solution_n, solution_np;

  // current time and time step size
  double time, time_step;

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  // use adaptive time stepping?
  bool const adaptive_time_stepping;

  unsigned int const n_refine_time;
  double const       cfl_number;
  double const       diffusion_number;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
