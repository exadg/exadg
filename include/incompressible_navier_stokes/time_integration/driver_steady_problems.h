/*
 * driver_steady_problems.h
 *
 *  Created on: Jul 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace IncNS
{
// forward declarations
template<int dim>
class InputParameters;

namespace Interface
{
template<int dim, typename Number>
class OperatorBase;
template<typename Number>
class OperatorCoupled;

} // namespace Interface

template<int dim, typename Number>
class DriverSteadyProblems
{
public:
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef Interface::OperatorBase<dim, Number> OperatorBase;
  typedef Interface::OperatorCoupled<Number>   OperatorPDE;

  DriverSteadyProblems(std::shared_ptr<OperatorBase> operator_base_in,
                       std::shared_ptr<OperatorPDE>  operator_in,
                       InputParameters<dim> const &  param_in);

  void
  setup();

  void
  solve_steady_problem();

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  solve();

  void
  postprocessing();

  std::shared_ptr<OperatorBase> operator_base;
  std::shared_ptr<OperatorPDE>  pde_operator;

  InputParameters<dim> const & param;

  std::vector<double> computing_times;

  ConditionalOStream pcout;

  BlockVectorType solution;
  BlockVectorType rhs_vector;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */
