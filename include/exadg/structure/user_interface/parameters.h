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

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_INPUT_PARAMETERS_H_

// ExaDG
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid_data.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver_data.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/structure/user_interface/enum_types.h>
#include <exadg/time_integration/enum_types.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/time_integration/solver_info_data.h>

namespace ExaDG
{
namespace Structure
{
class Parameters
{
public:
  // standard constructor that initializes parameters with default values
  Parameters();

  void
  check() const;

  bool
  involves_h_multigrid() const;

  void
  print(dealii::ConditionalOStream const & pcout, std::string const & name) const;

private:
  void
  print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_solver(dealii::ConditionalOStream const & pcout) const;

public:
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  ProblemType problem_type;

  // set true in order to consider body forces
  bool body_force;

  // are large deformations to be expected, than compute with non linear method
  bool large_deformation;

  // For nonlinear problems with large deformations, it is important to specify whether
  // the body forces are formulated with respect to the current deformed configuration
  // or the reference configuration.
  //
  // Option 1: pull_back_body_force = false
  // In this case, the body force is specified as force per undeformed volume. A typical
  // use case are density-proportional forces such as the gravitational force. The body
  // is then directly described in reference space, b_0 = rho_0 * g, and the pull-back to
  // the reference configuration is deactivated.
  //
  // Option 2: pull_back_body_force = true
  // The body force is specified as force per deformed volume, and the body force needs
  // to be pulled-back according to b_0 = dv/dV * b, where the volume ratio dv/dV depends
  // on the current state of deformation.
  bool pull_back_body_force;

  // For nonlinear problems with large deformations, it is important to specify whether
  // the traction Neumann boundary condition is formulated with respect to the current
  // deformed configuration or the reference configuration. Both cases appear in practice,
  // so it needs to be specified by the user which formulation is to be used.
  //
  // Option 1: pull_back_traction = false
  // In this case, the traction is specified as a force per undeformed area, e.g., a
  // force of fixed amount distributed uniformly over a surface of the body. The force per
  // deformed area is an unknown. Hence, it is more natural to specify the traction in the
  // reference configuration and deactivate the pull-back from the current to the reference
  // configuration.
  //
  // Option 2: pull_back_traction = true
  // The traction is known as a force per area of the deformed body. In this case, the
  // traction needs to be pulled-back to the reference configuration, i.e., t_0 = da/dA * t,
  // where the surface area ratio da/dA depends on the current state of deformation.
  // A typical use case would be fluid-structure-interaction problems where the fluid
  // stresses are applied as traction boundary conditions for the structure. Note that
  // the direction of the traction vector does not change by this pull-back operation.
  bool pull_back_traction;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/

  // density rho_0 in initial configuration (only relevant for unsteady problems)
  double density;

  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  double       start_time;
  double       end_time;
  double       time_step_size;
  unsigned int max_number_of_time_steps;

  // number of refinements for temporal discretization
  unsigned int n_refine_time;

  GenAlphaType gen_alpha_type;

  // spectral radius rho_infty for generalized alpha time integration scheme
  double spectral_radius;

  // configure printing of solver performance (wall time, number of iterations)
  SolverInfoData solver_info_data;

  // set this variable to true to start the simulation from restart files
  bool restarted_simulation;

  // restart
  RestartData restart_data;

  // quasi-static solver

  // choose a value in [0,1] where 1 = maximum load (Neumann or Dirichlet)
  double load_increment;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Grid data
  GridData grid;

  // Mapping
  unsigned int mapping_degree;

  // polynomial degree of shape functions
  unsigned int degree;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/

  // Newton solver data (only relevant for nonlinear problems)
  Newton::SolverData newton_solver_data;

  // description: see enum declaration
  Solver solver;

  // solver data
  SolverData solver_data;

  // description: see enum declaration
  Preconditioner preconditioner;

  // Applies to time-dependent OR nonlinear problems: update of preconditioner

  // Should the preconditioner be updated at all (set to false to never update the
  // preconditioner)?
  bool update_preconditioner;
  // If the above option is set to true, one can specify in more detail when to update
  // the preconditioner exactly:
  // - every ... time steps (or load steps for QuasiStatic problems)
  unsigned int update_preconditioner_every_time_steps;
  // and within a time step or load step:
  // - every ... Newton iterations (first update is invoked in the first Newton iteration)
  unsigned int update_preconditioner_every_newton_iterations;
  // - or once the Newton solver converged successfully (this option is currently used
  // in order to avoid invalid deformation states in non-converged Newton iterations)
  bool update_preconditioner_once_newton_converged;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;
};

} // namespace Structure
} // namespace ExaDG

#endif
