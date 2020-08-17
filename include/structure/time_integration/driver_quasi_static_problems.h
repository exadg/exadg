/*
 * driver_quasi_static_problems.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_TIME_INTEGRATION_DRIVER_QUASI_STATIC_PROBLEMS_H_
#define INCLUDE_STRUCTURE_TIME_INTEGRATION_DRIVER_QUASI_STATIC_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "utilities/timings_hierarchical.h"

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
class DriverQuasiStatic
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  DriverQuasiStatic(std::shared_ptr<Interface::Operator<Number>> operator_in,
                    std::shared_ptr<PostProcessorBase<Number>>   postprocessor_in,
                    InputParameters const &                      param_in,
                    MPI_Comm const &                             mpi_comm_in);

  void
  setup();

  void
  solve_problem();

  void
  print_iterations() const;

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
  output_solver_info_header(double const load_factor);

  std::tuple<unsigned int, unsigned int>
  solve_step(double const load_factor);

  void
  postprocessing() const;

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<Number>> postprocessor;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  // vectors
  VectorType solution;
  VectorType rhs_vector;

  unsigned int step_number;

  std::shared_ptr<TimerTree> timer_tree;

  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear}*/>
    iterations;
};

} // namespace Structure
} // namespace ExaDG

#endif
