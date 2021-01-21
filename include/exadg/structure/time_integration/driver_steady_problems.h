/*
 * driver_steady_problems.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

// forward declarations
class InputParameters;

template<typename Number>
class PostProcessorBase;

namespace Interface
{
template<typename Number>
class Operator;
}

template<int dim, typename Number>
class DriverSteady
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  DriverSteady(std::shared_ptr<Interface::Operator<Number>> operator_in,
               std::shared_ptr<PostProcessorBase<Number>>   postprocessor_in,
               InputParameters const &                      param_in,
               MPI_Comm const &                             mpi_comm_in,
               bool const                                   print_wall_times_in);

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

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<Number>> postprocessor;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  bool print_wall_times;

  ConditionalOStream pcout;

  // vectors
  VectorType solution;
  VectorType rhs_vector;

  std::shared_ptr<TimerTree> timer_tree;
};

} // namespace Structure
} // namespace ExaDG

#endif
