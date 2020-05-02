/*
 * driver_steady_problems.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../../utilities/timings_hierarchical.h"

using namespace dealii;

namespace ConvDiff
{
// forward declaration
class InputParameters;

template<typename Number>
class PostProcessorInterface;

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

  DriverSteadyProblems(std::shared_ptr<Operator>                       operator_in,
                       InputParameters const &                         param_in,
                       MPI_Comm const &                                mpi_comm_in,
                       std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  setup();

  void
  solve_problem();

  std::shared_ptr<TimerTree>
  get_timings() const;

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

  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  std::shared_ptr<TimerTree> timer_tree;

  // vectors
  VectorType solution;
  VectorType rhs_vector;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_ */
