/*
 * driver_steady_problems.h
 *
 *  Created on: Jul 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

// forward declarations
class InputParameters;

template<int dim, typename Number>
class DGNavierStokesCoupled;

template<typename Number>
class PostProcessorInterface;

template<int dim, typename Number>
class DriverSteadyProblems
{
public:
  typedef LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> Operator;

  DriverSteadyProblems(std::shared_ptr<Operator>                       operator_in,
                       InputParameters const &                         param_in,
                       MPI_Comm const &                                mpi_comm_in,
                       std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  setup();

  void
  solve_steady_problem();

  VectorType const &
  get_velocity() const;

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

  std::shared_ptr<TimerTree> timer_tree;

  ConditionalOStream pcout;

  BlockVectorType solution;
  BlockVectorType rhs_vector;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */
