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

#ifndef INCLUDE_EXADG_POISSON_DRIVER_H_
#define INCLUDE_EXADG_POISSON_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/application_base.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Poisson
{
enum class OperatorType
{
  Evaluate,
  Apply
};

template<int dim, int n_components, typename Number>
class Solver
{
public:
  void
  setup(std::shared_ptr<ApplicationBase<dim, n_components, Number>> application,
        MPI_Comm const                                              mpi_comm,
        bool const                                                  is_throughput_study)
  {
    pde_operator =
      std::make_shared<Operator<dim, n_components, Number>>(application->get_grid(),
                                                            application->get_mapping(),
                                                            application->get_boundary_descriptor(),
                                                            application->get_field_functions(),
                                                            application->get_parameters(),
                                                            "Poisson",
                                                            mpi_comm);

    pde_operator->setup();

    if(not(is_throughput_study))
    {
      postprocessor = application->create_postprocessor();
      postprocessor->setup(*pde_operator);
    }
  }

  std::shared_ptr<Operator<dim, n_components, Number>>          pde_operator;
  std::shared_ptr<PostProcessorBase<dim, n_components, Number>> postprocessor;
};

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const &                                 mpi_comm,
         std::shared_ptr<ApplicationBase<dim, 1, Number>> application,
         bool const                                       is_test,
         bool const                                       is_throughput_study);

  void
  setup();

  void
  solve();

  SolverResult
  print_performance_results(double const total_time) const;

  /*
   * Throughput study
   */
  std::tuple<unsigned int, dealii::types::global_dof_index, double>
  apply_operator(OperatorType const & operator_type,
                 unsigned int const   n_repetitions_inner,
                 unsigned int const   n_repetitions_outer) const;

private:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // do not set up certain data structures (solver, postprocessor) in case of throughput study
  bool const is_throughput_study;

  // application
  std::shared_ptr<ApplicationBase<dim, 1, Number>> application;

  std::shared_ptr<Solver<dim, 1, Number>> poisson;

  // number of iterations
  mutable unsigned int iterations;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
  mutable double    solve_time;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_DRIVER_H_ */
