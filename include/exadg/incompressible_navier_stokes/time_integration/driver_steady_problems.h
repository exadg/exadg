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
class OperatorCoupled;

template<typename Number>
class PostProcessorInterface;

template<int dim, typename Number>
class DriverSteadyProblems
{
public:
  typedef LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> Operator;

  DriverSteadyProblems(std::shared_ptr<Operator>                       operator_in,
                       InputParameters const &                         param_in,
                       MPI_Comm const &                                mpi_comm_in,
                       bool const                                      print_wall_times_in,
                       std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  setup();

  void
  solve_steady_problem(double const time = 0.0, bool unsteady_problem = false);

  VectorType const &
  get_velocity() const;

  std::shared_ptr<TimerTree>
  get_timings() const;

  void
  print_iterations() const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  solve(double const time = 0.0, bool unsteady_problem = false);

  bool
  print_solver_info(double const time, bool unsteady_problem = false) const;

  void
  postprocessing(double const time = 0.0, bool unsteady_problem = false) const;

  std::shared_ptr<Operator> pde_operator;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  bool print_wall_times;

  Timer                      global_timer;
  std::shared_ptr<TimerTree> timer_tree;

  ConditionalOStream pcout;

  BlockVectorType solution;
  BlockVectorType rhs_vector;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
    iterations;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */
