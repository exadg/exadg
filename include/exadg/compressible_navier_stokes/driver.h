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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_

#include <exadg/compressible_navier_stokes/spatial_discretization/operator.h>
#include <exadg/compressible_navier_stokes/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/compressible_navier_stokes/user_interface/analytical_solution.h>
#include <exadg/compressible_navier_stokes/user_interface/application_base.h>
#include <exadg/compressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/compressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/compressible_navier_stokes/user_interface/parameters.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace CompNS
{
// Select the operator to be applied
enum class OperatorType
{
  ConvectiveTerm,
  ViscousTerm,
  ViscousAndConvectiveTerms,
  InverseMassOperator,
  InverseMassOperatorDstDst,
  VectorUpdate,
  EvaluateOperatorExplicit
};

inline unsigned int
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)input_file;

  unsigned int const dofs_per_element = (dim + 2) * dealii::Utilities::pow(degree + 1, dim);

  return dofs_per_element;
}

template<int dim, typename Number = double>
class Driver
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  Driver(MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test,
         bool const                                    is_throughput_study);

  void
  setup();

  void
  solve();

  void
  print_performance_results(double const total_time) const;

  /*
   * Throughput study
   */
  std::tuple<unsigned int, dealii::types::global_dof_index, double>
  apply_operator(std::string const & operator_type,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

private:
  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // do not set up certain data structures (solver, postprocessor) in case of throughput study
  bool const is_throughput_study;

  std::shared_ptr<ApplicationBase<dim, Number>> application;

  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  std::shared_ptr<Operator<dim, Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  std::shared_ptr<TimeIntExplRK<Number>> time_integrator;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_ */
