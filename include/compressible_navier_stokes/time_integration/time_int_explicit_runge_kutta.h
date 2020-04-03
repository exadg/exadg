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

#include "time_integration/time_int_explicit_runge_kutta_base.h"

#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/ssp_runge_kutta.h"

#include "../postprocessor/postprocessor_base.h"

using namespace dealii;

namespace CompNS
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class Operator;
}

template<typename Number>
class TimeIntExplRK : public TimeIntExplRKBase<Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  TimeIntExplRK(std::shared_ptr<Operator>                       operator_in,
                InputParameters const &                         param_in,
                unsigned int const                              refine_steps_time_in,
                MPI_Comm const &                                mpi_comm_in,
                std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  initialize_time_integrator();

  void
  initialize_vectors();

  void
  initialize_solution();

  void
  detect_instabilities() const;

  void
  postprocessing() const;

  void
  solve_timestep();

  bool
  print_solver_info() const;

  void
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

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

  InputParameters const & param;

  unsigned int const refine_steps_time;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // timer
  mutable Timer  timer_postprocessing;
  mutable double time_postprocessing;

  // monitor the L2-norm of the solution vector in order to detect instabilities
  mutable double l2_norm;

  // time step calculation
  double const cfl_number;
  double const diffusion_number;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
