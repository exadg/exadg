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

#ifndef INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_QUASI_STATIC_PROBLEMS_H_
#define INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_DRIVER_QUASI_STATIC_PROBLEMS_H_

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
class Parameters;

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
  DriverQuasiStatic(std::shared_ptr<Interface::Operator<Number>> operator_,
                    std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
                    Parameters const &                           param_,
                    MPI_Comm const &                             mpi_comm_,
                    bool const                                   is_test_);

  void
  setup();

  void
  solve();

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
  do_solve();

  void
  output_solver_info_header(double const load_factor);

  std::tuple<unsigned int, unsigned int>
  solve_step(double const load_factor);

  void
  postprocessing() const;

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<Number>> postprocessor;

  Parameters const & param;

  MPI_Comm const mpi_comm;

  bool const is_test;

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
