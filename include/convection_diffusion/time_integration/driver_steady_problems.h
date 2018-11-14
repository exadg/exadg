/*
 * driver_steady_problems.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace ConvDiff
{
// forward declaration
class InputParameters;

namespace Interface
{
template<typename Number>
class Operator;
}

template<typename Number>
class DriverSteadyProblems
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  DriverSteadyProblems(std::shared_ptr<Operator> operator_in, InputParameters const & param_in);
  void
  setup();

  void
  solve_problem();

  void
  analyze_computing_times() const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  solve();

  void
  postprocessing() const;

  std::shared_ptr<Operator> pde_operator;

  InputParameters const & param;

  // timer
  Timer  global_timer;
  double total_time;

  // vectors
  VectorType solution;
  VectorType rhs_vector;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_ */
