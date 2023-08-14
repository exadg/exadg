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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_H_

#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/mapping_deformation_function.h>
#include <exadg/grid/mapping_deformation_poisson.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_base.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/driver_steady_problems.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/user_interface/application_base.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/finite_element.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace IncNS
{
// Note: Make sure that the correct time integration scheme is selected in the input file that is
//       compatible with the OperatorType specified here. This also includes the treatment of the
//       convective term (explicit/implicit), e.g., specifying VelocityConvDiffOperator together
//       with an explicit treatment of the convective term will only apply the Helmholtz-like
//       operator.

// clang-format off
enum class OperatorType{
  CoupledNonlinearResidual, // nonlinear residual of coupled system of equations
  CoupledLinearized,        // linearized system of equations for coupled solution approach
  PressurePoissonOperator,  // negative Laplace operator (scalar quantity, pressure)
  ConvectiveOperator,       // convective term (vectorial quantity, velocity)
  HelmholtzOperator,        // mass + viscous (vectorial quantity, velocity)
  ProjectionOperator,       // mass + divergence penalty + continuity penalty (vectorial quantity, velocity)
  VelocityConvDiffOperator, // mass + convective + viscous (vectorial quantity, velocity)
  InverseMassOperator       // inverse mass operator (vectorial quantity, velocity)
};
// clang-format on

enum class PressureDegree
{
  MixedOrder,
  EqualOrder
};

inline unsigned int
get_dofs_per_element(OperatorType const &     operator_type,
                     PressureDegree const &   pressure_degree,
                     unsigned int const       dim,
                     unsigned int const       degree,
                     ExaDG::ElementType const element_type)
{
  unsigned int degree_p = 1;
  if(pressure_degree == PressureDegree::MixedOrder)
    degree_p = degree - 1;
  else if(pressure_degree == PressureDegree::EqualOrder)
    degree_p = degree;
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));

  unsigned int const velocity_dofs_per_element = ExaDG::get_dofs_per_element(
    element_type, true /* is_dg */, dim /* n_components */, degree, dim);

  unsigned int const pressure_dofs_per_element = ExaDG::get_dofs_per_element(
    element_type, true /* is_dg */, 1 /* n_components */, degree_p, dim);

  // coupled/monolithic problem
  if(operator_type == OperatorType::CoupledNonlinearResidual or
     operator_type == OperatorType::CoupledLinearized)
  {
    return velocity_dofs_per_element + pressure_dofs_per_element;
  }
  // velocity only
  else if(operator_type == OperatorType::ConvectiveOperator or
          operator_type == OperatorType::VelocityConvDiffOperator or
          operator_type == OperatorType::HelmholtzOperator or
          operator_type == OperatorType::ProjectionOperator or
          operator_type == OperatorType::InverseMassOperator)
  {
    return velocity_dofs_per_element;
  }
  // pressure only
  else if(operator_type == OperatorType::PressurePoissonOperator)
  {
    return pressure_dofs_per_element;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  return 0;
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

  // moving mapping (ALE)
  std::shared_ptr<DeformedMappingBase<dim, Number>> grid_motion;

  // ALE helper functions required by time integrator
  std::shared_ptr<HelpersALE<Number>> helpers_ale;

  /*
   * Spatial discretization
   */
  std::shared_ptr<SpatialOperatorBase<dim, Number>> pde_operator;

  /*
   * Postprocessor
   */
  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  /*
   * Temporal discretization
   */

  // unsteady solver
  std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator;

  // steady solver
  std::shared_ptr<DriverSteadyProblems<dim, Number>> driver_steady;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_H_ */
