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
#include "time_integration/time_int_explicit_runge_kutta_base.h"

#include "../postprocessor/postprocessor_base.h"

using namespace dealii;

namespace ConvDiff
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class Operator;

template<typename Number>
class OperatorExplRK;
} // namespace Interface

template<typename Number>
class TimeIntExplRK : public TimeIntExplRKBase<Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number>       Operator;
  typedef Interface::OperatorExplRK<Number> ExplRKOperator;

  TimeIntExplRK(std::shared_ptr<Operator>                       operator_in,
                InputParameters const &                         param_in,
                MPI_Comm const &                                mpi_comm_in,
                std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in);

  void
  extrapolate_solution(VectorType & vector);

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  postprocessing() const;

  bool
  print_solver_info() const;

  void
  solve_timestep();

  void
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  void
  initialize_time_integrator();

  std::shared_ptr<Operator> pde_operator;

  std::shared_ptr<ExplRKOperator> expl_rk_operator;

  std::shared_ptr<ExplicitTimeIntegrator<ExplRKOperator, VectorType>> rk_time_integrator;

  InputParameters const & param;

  std::vector<VectorType const *> velocities;
  std::vector<double>             times;

  // store time step size according to diffusion condition so that it does not have to be
  // recomputed in case of adaptive time stepping
  double time_step_diff;

  double const cfl;
  double const diffusion_number;

  double wall_time;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
