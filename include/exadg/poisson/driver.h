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
#include <exadg/poisson/solver_poisson.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/application_base.h>
#include <exadg/utilities/solver_result.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Poisson
{
enum class OperatorType
{
  MatrixFree,
  MatrixBased
};

inline std::string
enum_to_string(OperatorType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case OperatorType::MatrixFree:  string_type = "MatrixFree";  break;
    case OperatorType::MatrixBased: string_type = "MatrixBased"; break;
    default: AssertThrow(false, dealii::ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(OperatorType & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "MatrixFree")  enum_type = OperatorType::MatrixFree;
  else if(string_type == "MatrixBased") enum_type = OperatorType::MatrixBased;
  else AssertThrow(false, dealii::ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  std::string spatial_discretization = "DG";

  dealii::ParameterHandler prm;
  prm.enter_subsection("Discretization");
  prm.add_parameter("SpatialDiscretization",
                    spatial_discretization,
                    "Spatial discretization (CG vs. DG).",
                    dealii::Patterns::Selection("CG|DG"),
                    true);
  prm.leave_subsection();

  prm.parse_input(input_file, "", true, true);

  unsigned int dofs_per_element = 1;

  if(spatial_discretization == "CG")
    dofs_per_element = dealii::Utilities::pow(degree, dim);
  else if(spatial_discretization == "DG")
    dofs_per_element = dealii::Utilities::pow(degree + 1, dim);
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));

  return dofs_per_element;
}

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
  apply_operator(std::string const & operator_type_string,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

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

  std::shared_ptr<SolverPoisson<dim, 1, Number>> poisson;

  // number of iterations
  mutable unsigned int iterations;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
  mutable double    solve_time;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_DRIVER_H_ */
