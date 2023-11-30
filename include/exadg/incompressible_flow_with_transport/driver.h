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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_

// application
#include <exadg/incompressible_flow_with_transport/user_interface/application_base.h>

// utilities
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

// ConvDiff
#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>
#include <exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h>

// IncNS
#include <exadg/convection_diffusion/spatial_discretization/operator.h>
#include <exadg/grid/mapping_deformation_function.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/driver_steady_problems.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

namespace ExaDG
{
namespace FTI
{
template<int dim, typename Number = double>
class Driver
{
public:
  Driver(MPI_Comm const &                              mpi_comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test);

  void
  setup();

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

private:
  void
  ale_update() const;

  void
  communicate_scalar_to_fluid() const;

  void
  communicate_fluid_to_all_scalars() const;

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

  // grid motion (ALE)
  std::shared_ptr<DeformedMappingFunction<dim, Number>> ale_mapping;

  // ALE helper functions required by time integrator
  std::shared_ptr<HelpersALE<Number>> helpers_ale;

  bool use_adaptive_time_stepping;

  //  MatrixFree (only a single object for both flow and transport problems)
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  // INCOMPRESSIBLE NAVIER-STOKES

  std::shared_ptr<IncNS::SpatialOperatorBase<dim, Number>> fluid_operator;

  typedef IncNS::PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> fluid_postprocessor;

  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> fluid_time_integrator;

  // steady solver
  typedef IncNS::DriverSteadyProblems<dim, Number> DriverSteady;

  std::shared_ptr<DriverSteady> fluid_driver_steady;

  // SCALAR TRANSPORT

  std::vector<std::shared_ptr<ConvDiff::Operator<dim, Number>>> scalar_operator;

  std::vector<std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>> scalar_postprocessor;

  std::vector<std::shared_ptr<TimeIntBase>> scalar_time_integrator;

  mutable dealii::LinearAlgebra::distributed::Vector<Number> temperature;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;

  mutable unsigned int N_time_steps;
};

} // namespace FTI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_ */
