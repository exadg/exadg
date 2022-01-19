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

#ifndef INCLUDE_EXADG_STRUCTURE_DRIVER_H_
#define INCLUDE_EXADG_STRUCTURE_DRIVER_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/driver_quasi_static_problems.h>
#include <exadg/structure/time_integration/driver_steady_problems.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>
#include <exadg/structure/user_interface/application_base.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

enum class OperatorType
{
  Nonlinear,
  Linearized
};

inline std::string
enum_to_string(OperatorType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case OperatorType::Nonlinear:  string_type = "Nonlinear";  break;
    case OperatorType::Linearized: string_type = "Linearized"; break;
    default: AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(OperatorType & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "Nonlinear")  enum_type = OperatorType::Nonlinear;
  else if(string_type == "Linearized") enum_type = OperatorType::Linearized;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)input_file;

  unsigned int const dofs_per_element = Utilities::pow(degree, dim) * dim;

  return dofs_per_element;
}

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test,
         bool const                                    is_throughput_study);

  void
  setup();

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

  /*
   * Throughput study
   */
  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(std::string const & operator_type_string,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

private:
  // MPI communicator
  MPI_Comm mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // do not set up certain data structures (solver, postprocessor) in case of throughput study
  bool const is_throughput_study;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // user parameters
  Parameters param;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;

  // operator
  std::shared_ptr<Operator<dim, Number>> pde_operator;

  // postprocessor
  std::shared_ptr<PostProcessor<dim, Number>> postprocessor;

  // driver steady-state
  std::shared_ptr<DriverSteady<dim, Number>> driver_steady;

  // driver quasi-static
  std::shared_ptr<DriverQuasiStatic<dim, Number>> driver_quasi_static;

  // time integration scheme
  std::shared_ptr<TimeIntGenAlpha<dim, Number>> time_integrator;

  // computation time
  mutable TimerTree timer_tree;
};

} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_EXADG_STRUCTURE_DRIVER_H_ */
