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

#ifndef EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Structure
{
// forward declarations
class Parameters;

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
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  DriverSteady(std::shared_ptr<Interface::Operator<Number>> operator_,
               std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
               Parameters const &                           param_,
               MPI_Comm const &                             mpi_comm_,
               bool const                                   is_test_);

  void
  setup();

  void
  solve();

  std::shared_ptr<TimerTree>
  get_timings() const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  do_solve();

  void
  postprocessing() const;

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<Number>> postprocessor;

  Parameters const & param;

  MPI_Comm const mpi_comm;

  bool is_test;

  dealii::ConditionalOStream pcout;

  // vectors
  VectorType solution;
  VectorType rhs_vector;

  std::shared_ptr<TimerTree> timer_tree;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */
