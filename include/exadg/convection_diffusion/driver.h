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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_DRIVER_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// ExaDG
#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/spatial_discretization/interface.h>
#include <exadg/convection_diffusion/spatial_discretization/operator.h>
#include <exadg/convection_diffusion/time_integration/driver_steady_problems.h>
#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>
#include <exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/convection_diffusion/user_interface/analytical_solution.h>
#include <exadg/convection_diffusion/user_interface/application_base.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/mapping_deformation_function.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace ConvDiff
{
enum class OperatorType
{
  MassOperator,
  ConvectiveOperator,
  DiffusiveOperator,
  MassConvectionDiffusionOperator
};

inline unsigned int
get_dofs_per_element(unsigned int const       dim,
                     unsigned int const       degree,
                     ExaDG::ElementType const element_type)
{
  // TODO
  (void)element_type;

  unsigned int const dofs_per_element = dealii::Utilities::pow(degree + 1, dim);

  return dofs_per_element;
}

template<int dim, typename Number = double>
class Driver
{
public:
  Driver(MPI_Comm const &                              mpi_comm,
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
  apply_operator(OperatorType const & operator_type,
                 unsigned int const   n_repetitions_inner,
                 unsigned int const   n_repetitions_outer) const;

private:
  void
  ale_update() const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // do not set up certain data structures (solver, postprocessor) in case of throughput study
  bool const is_throughput_study;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // ALE mapping
  std::shared_ptr<DeformedMappingFunction<dim, Number>> ale_mapping;

  // ALE helper functions required by time integrator
  std::shared_ptr<HelpersALE<Number>> helpers_ale;

  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  std::shared_ptr<Operator<dim, Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  std::shared_ptr<TimeIntBase> time_integrator;

  std::shared_ptr<DriverSteadyProblems<Number>> driver_steady;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_DRIVER_H_ */
