/*
 * time_int_explicit_runge_kutta.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/ssp_runge_kutta.h"

using namespace dealii;

namespace CompNS
{
// forward declarations
template<int dim>
class InputParameters;

namespace Interface
{
template<typename Number>
class Operator;
}

template<int dim, typename Number>
class TimeIntExplRK
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  TimeIntExplRK(std::shared_ptr<Operator>    operator_in,
                InputParameters<dim> const & param_in,
                unsigned int const           n_refine_time_in);

  void
  timeloop();

  void
  setup();

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  detect_instabilities();

  void
  postprocessing();

  void
  solve_timestep();

  void
  prepare_vectors_for_next_timestep();

  void
  calculate_timestep();

  void
  analyze_computing_times() const;

  void
  calculate_pressure();

  void
  calculate_velocity();

  void
  calculate_temperature();

  void
  calculate_vorticity();

  void
  calculate_divergence();

  std::shared_ptr<Operator> pde_operator;

  std::shared_ptr<ExplicitTimeIntegrator<Operator, VectorType>> rk_time_integrator;

  InputParameters<dim> const & param;

  // timer
  Timer  global_timer, timer_postprocessing;
  double total_time;
  double time_postprocessing;

  // screen output
  ConditionalOStream pcout;

  // monitor the L2-norm of the solution vector in order to detect instabilities
  double l2_norm;

  // DoF vectors for conserved variables: (rho, rho u, rho E)
  VectorType solution_n, solution_np;

  // current time and time step size
  double time, time_step;

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  // time refinement steps
  unsigned int const n_refine_time;

  // time step calculation
  double const cfl_number;
  double const diffusion_number;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
