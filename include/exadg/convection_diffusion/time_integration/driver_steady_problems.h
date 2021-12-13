/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

// forward declaration
class Parameters;

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

  DriverSteadyProblems(std::shared_ptr<Operator>                       operator_,
                       Parameters const &                              param_,
                       MPI_Comm const &                                mpi_comm_,
                       bool const                                      is_test_,
                       std::shared_ptr<PostProcessorInterface<Number>> postprocessor_);

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

  Parameters const & param;

  MPI_Comm const mpi_comm;

  bool is_test;

  ConditionalOStream pcout;

  std::shared_ptr<TimerTree> timer_tree;

  // vectors
  VectorType solution;
  VectorType rhs_vector;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_DRIVER_STEADY_PROBLEMS_H_ */
