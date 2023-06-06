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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid_data.h>
#include <exadg/incompressible_navier_stokes/user_interface/enum_types.h>
#include <exadg/incompressible_navier_stokes/user_interface/viscosity_model_data.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver_data.h>
#include <exadg/solvers_and_preconditioners/preconditioners/enum_types.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/time_integration/enum_types.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/time_integration/solver_info_data.h>

namespace ExaDG
{
namespace IncNS
{
class Parameters
{
public:
  // standard constructor that initializes parameters
  Parameters();

  void
  check(dealii::ConditionalOStream const & pcout) const;

  bool
  convective_problem() const;

  bool
  viscous_problem() const;

  bool
  viscous_term_is_nonlinear() const;

  bool
  viscosity_is_variable() const;

  bool
  implicit_convective_problem() const;

  bool
  nonlinear_viscous_problem() const;

  bool
  nonlinear_problem_has_to_be_solved() const;

  bool
  involves_h_multigrid() const;

  unsigned int
  get_degree_p(unsigned int const degree_u) const;

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
  print_parameters_numerical_parameters(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_pressure_poisson(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_projection_step(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_dual_splitting(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_pressure_correction(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_coupled_solver(dealii::ConditionalOStream const & pcout) const;

  // coupled solver
  bool
  involves_h_multigrid_velocity_block() const;

  bool
  involves_h_multigrid_pressure_block() const;


  // penalty step (coupled solver or projection methods)
  bool
  involves_h_multigrid_penalty_step() const;

  // projection methods
  bool
  involves_h_multigrid_pressure_step() const;

  bool
  involves_h_multigrid_viscous_step() const;

  bool
  involves_h_multigrid_momentum_step() const;

public:
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  ProblemType problem_type;

  // description: see enum declaration
  EquationType equation_type;

  // description: see enum declaration
  FormulationViscousTerm formulation_viscous_term;

  // description: see enum declaration
  FormulationConvectiveTerm formulation_convective_term;

  // use stable outflow boundary condition for convective term according to
  // Gravemeier et al. (2012)
  bool use_outflow_bc_convective_term;

  // if the body force vector on the right-hand side of the momentum equation of the
  // Navier-Stokes equations is unequal zero, set right_hand_side = true
  // This parameter also has to be true when considering the Boussinesq term.
  bool right_hand_side;

  // Boussinesq term (natural convection through buoyancy forces)
  bool boussinesq_term;

  // If Boussinesq term is activated: solves only for dynamic pressure variations if true,
  // and includes hydrostatic component if false.
  bool boussinesq_dynamic_part_only;

  /**************************************************************************************/
  /*                                                                                    */
  /*                 Arbitrary Lagrangian-Eulerian formulation (ALE)                    */
  /*                                                                                    */
  /**************************************************************************************/

  // use true to activate ALE formulation, otherwise the standard Eulerian formulation
  // with fixed mesh will be used
  bool ale_formulation;

  MeshMovementType mesh_movement_type;

  bool neumann_with_variable_normal_vector;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/

  // start time of simulation
  double start_time;

  // end time of simulation
  double end_time;

  // kinematic viscosity
  double viscosity;

  // density (not required by fluid solver which is formulated in terms of the kinematic
  // viscosity only, but for the calculation of the fluid stress for FSI problems)
  double density;

  // Boussinesq term
  double thermal_expansion_coefficient;
  double reference_temperature;

  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  SolverType solver_type;

  // description: see enum declaration
  TemporalDiscretization temporal_discretization;

  // description: see enum declaration
  TreatmentOfConvectiveTerm treatment_of_convective_term;

  // description: see enum declaration
  TimeStepCalculation calculation_of_time_step_size;

  // use adaptive time stepping?
  bool adaptive_time_stepping;

  // This parameter defines by which factor the time step size is allowed to increase
  // or to decrease in case of adaptive time step, e.g., if one wants to avoid large
  // jumps in the time step size. A factor of 1 implies that the time step size can not
  // change at all, while a factor towards infinity implies that arbitrary changes in
  // the time step size are allowed from one time step to the next.
  double adaptive_time_stepping_limiting_factor;

  // specify a maximum time step size in case of adaptive time stepping since the adaptive
  // time stepping algorithm would choose arbitrarily large time step sizes if the velocity field
  // is close to zero. This variable is only used for adaptive time stepping.
  double time_step_size_max;

  // Different variants are available for calculating the time step size based on a local CFL
  // criterion.
  CFLConditionType adaptive_time_stepping_cfl_type;

  // maximum velocity needed when calculating the time step according to cfl-condition
  double max_velocity;

  // cfl number: note that this cfl number is the first in a series of cfl numbers
  // when performing temporal convergence tests, i.e., cfl_real = cfl, cfl/2, cfl/4, ...
  // ("global" CFL number, can be larger than critical CFL in case
  // of operator-integration-factor splitting)
  double cfl;

  // dt = CFL/k_u^{exp} * h / || u ||
  double cfl_exponent_fe_degree_velocity;

  // C_eff: constant that has to be specified for time step calculation method
  // MaxEfficiency, which means that the time step is selected such that the errors of
  // the temporal and spatial discretization are comparable
  double c_eff;

  // user specified time step size:  note that this time_step_size is the first
  // in a series of time_step_size's when performing temporal convergence tests,
  // i.e., delta_t = time_step_size, time_step_size/2, ...
  double time_step_size;

  // maximum number of time steps
  unsigned int max_number_of_time_steps;

  // number of refinements for temporal discretization
  unsigned int n_refine_time;

  // order of BDF time integration scheme and extrapolation scheme
  unsigned int order_time_integrator;

  // start time integrator with low order time integrator, i.e., first order Euler method
  bool start_with_low_order;

  // description: see enum declaration
  ConvergenceCriterionSteadyProblem convergence_criterion_steady_problem;

  // Pseudo-timestepping for steady-state problems: These tolerances are only relevant
  // when using an unsteady solver to solve the steady Navier-Stokes equations.
  //
  // option ResidualNavierStokes:
  // - these tolerances refer to the norm of the residual of the steady
  //   Navier-Stokes equations.
  //
  // option SolutionIncrement:
  // - these tolerances refer to the norm of the increment of the solution
  //   vector from one time step to the next.
  double abs_tol_steady;
  double rel_tol_steady;

  // show solver performance (wall time, number of iterations) every ... timesteps
  SolverInfoData solver_info_data;

  // set this variable to true to start the simulation from restart files
  bool restarted_simulation;

  // restart
  RestartData restart_data;


  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Grid data
  GridData grid;

  // Mapping
  unsigned int mapping_degree;

  // type of spatial discretization approach
  SpatialDiscretization spatial_discretization;

  // Polynomial degree of velocity shape functions. In case of H-div Raviart-Thomas 'degree_u' is
  // the polynomial degree in normal direction and (degree_u - 1) is the degree in tangential
  // direction.
  unsigned int degree_u;

  // Polynomial degree of pressure shape functions
  DegreePressure degree_p;

  // convective term: upwind factor describes the scaling factor in front of the
  // stabilization term (which is strictly dissipative) of the numerical function
  // of the convective term. For the divergence formulation of the convective term with
  // local Lax-Friedrichs flux, a value of upwind_factor = 1.0 corresponds to the
  // theoretical value (e.g., maximum eigenvalue of the flux Jacobian, lambda = 2 |u*n|)
  // but a lower value (e.g., upwind_factor = 0.5, lambda = |u*n|) might be much more
  // advantages in terms of computational costs by allowing significantly larger time
  // step sizes.
  double upwind_factor;

  // description: see enum declaration
  TypeDirichletBCs type_dirichlet_bc_convective;

  // description: see enum declaration
  InteriorPenaltyFormulation IP_formulation_viscous;

  // description: see enum declaration
  PenaltyTermDivergenceFormulation penalty_term_div_formulation;

  // interior penalty parameter scaling factor for Helmholtz equation of viscous step
  double IP_factor_viscous;

  // integration by parts of grad(P)
  bool gradp_integrated_by_parts;

  // type for formulation
  FormulationPressureGradientTerm gradp_formulation;

  // use boundary data if integrated by parts
  bool gradp_use_boundary_data;

  // integration by parts of div(U)
  bool divu_integrated_by_parts;

  // type of formulation
  FormulationVelocityDivergenceTerm divu_formulation;

  // use boundary data if integrated by parts
  bool divu_use_boundary_data;

  // For certain setups and types of boundary conditions, the pressure level is undefined,
  // e.g., if the velocity is prescribed on the whole boundary or in case of periodic
  // boundary conditions. This variable defines the method used to adjust the pressure level
  // in or to obtain a well-defined pressure solution.
  // If you observe convergence in the velocity error, but no convergence in the pressure
  // error, this might likely be related to the fact that this parameter has not been set
  // as intended.
  AdjustPressureLevel adjust_pressure_level;

  // use div-div penalty term
  bool use_divergence_penalty;

  // penalty factor of divergence penalty term
  double divergence_penalty_factor;

  // use continuity penalty term
  bool use_continuity_penalty;

  // penalty factor of continuity penalty term
  double continuity_penalty_factor;

  // Divergence and continuity penalty terms are applied in a postprocessing step
  // if set to true.
  // Otherwise, penalty terms are added to the monolithic systems of equations in case of
  // the monolithic solver, or to the projection step in case of the dual splitting scheme.
  // For the pressure-correction scheme, this parameter is irrelevant since the projection
  // step is the last step of the splitting scheme anyway.
  bool apply_penalty_terms_in_postprocessing_step;

  // specify which components of the velocity field to penalize, i.e., normal
  // components only or all components
  ContinuityPenaltyComponents continuity_penalty_components;

  // specify whether boundary conditions prescribed for the velocity should be
  // used in continuity penalty operator. Otherwise, boundary faces are ignored.
  bool continuity_penalty_use_boundary_data;

  // type of penalty parameter (see enum declaration for more information)
  TypePenaltyParameter type_penalty_parameter;

  /**************************************************************************************/
  /*                                                                                    */
  /*                            Variable viscosity models                               */
  /*                                                                                    */
  /**************************************************************************************/

  TreatmentOfVariableViscosity  treatment_of_variable_viscosity;
  TurbulenceModelData           turbulence_model_data;
  GeneralizedNewtonianModelData generalized_newtonian_model_data;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              NUMERICAL PARAMETERS                                  */
  /*                                                                                    */
  /**************************************************************************************/

  // Implement block diagonal (block Jacobi) preconditioner in a matrix-free way
  // by solving the block Jacobi problems elementwise using iterative solvers and
  // matrix-free operator evaluation. By default, this variable should be set to true
  // because the matrix-based variant (which is used otherwise) is very slow and the
  // matrix-free variant can be expected to be much faster.
  // Only in case that convergence problems occur or for reasons of testing/debugging
  // the matrix-based variant should be used.
  bool implement_block_diagonal_preconditioner_matrix_free;

  // By default, the matrix-free implementation performs separate loops over all cells,
  // interior faces, and boundary faces. For a certain type of operations, however, it
  // is necessary to perform the face-loop as a loop over all faces of a cell with an
  // outer loop over all cells, e.g., preconditioners operating on the level of
  // individual cells (for example block Jacobi). With this parameter, the loop structure
  // can be changed to such an algorithm (cell_based_face_loops).
  bool use_cell_based_face_loops;

  // Solver data for block Jacobi preconditioner. Accordingly, this parameter is only
  // relevant if the block diagonal preconditioner is implemented in a matrix-free way
  // using an elementwise iterative solution procedure for which solver tolerances have to
  // be provided by the user. It was found that rather coarse relative solver  tolerances
  // of around 1.e-2 are enough and for lower tolerances one would 'over-solve' the local
  // preconditioning problems without a further benefit in global iteration counts.
  SolverData solver_data_block_diagonal;

  // Quadrature rule used to integrate the linearized convective term. This parameter is
  // therefore only relevant if linear systems of equations have to be solved involving
  // the convective term. For reasons of computational efficiency, it might be advantageous
  // to use a standard quadrature rule for the linearized problem in order to speed up
  // the computation. However, it was found that choosing a lower order quadrature rule
  // for the linearized problem only, increases the number of iterations significantly. It
  // was found that the quadrature rules used for the nonlinear and linear problems should
  // be the same. Hence, although this parameter speeds up the operator evaluation (i.e.
  // the wall time per iteration), it is unclear whether a lower order quadrature rule
  // really allows to achieve a more efficient method overall.
  QuadratureRuleLinearization quad_rule_linearization;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PROJECTION METHODS                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // PRESSURE POISSON EQUATION

  // interior penalty parameter scaling factor for pressure Poisson equation
  double IP_factor_pressure;

  // description: see enum declaration
  SolverPressurePoisson solver_pressure_poisson;

  // solver data for pressure Poisson equation
  SolverData solver_data_pressure_poisson;

  // description: see enum declaration
  PreconditionerPressurePoisson preconditioner_pressure_poisson;

  // update of preconditioner for this equation is currently not provided and not needed

  // description: see declaration of MultigridData
  MultigridData multigrid_data_pressure_poisson;

  // Update preconditioner before solving the linear system of equations.
  bool update_preconditioner_pressure_poisson;

  // Update preconditioner every ... time steps.
  // This variable is only used if update of preconditioner is true.
  unsigned int update_preconditioner_pressure_poisson_every_time_steps;

  // PROJECTION STEP

  // description: see enum declaration
  SolverProjection solver_projection;

  // solver data for projection step
  SolverData solver_data_projection;

  // description: see enum declaration
  PreconditionerProjection preconditioner_projection;

  // description: see declaration of MultigridData
  MultigridData multigrid_data_projection;

  // Update preconditioner before solving the linear system of equations.
  // Note that this variable is only used when using an iterative method
  // to solve the global projection equation.
  bool update_preconditioner_projection;

  // Update preconditioner every ... time steps.
  // This variable is only used if update of preconditioner is true.
  unsigned int update_preconditioner_projection_every_time_steps;

  // description: see enum declaration (only relevant if block diagonal is used as
  // preconditioner)
  Elementwise::Preconditioner preconditioner_block_diagonal_projection;

  // solver data for block Jacobi preconditioner (only relevant if elementwise
  // iterative solution procedure is used for block diagonal preconditioner)
  SolverData solver_data_block_diagonal_projection;

  /**************************************************************************************/
  /*                                                                                    */
  /*                        HIGH-ORDER DUAL SPLITTING SCHEME                            */
  /*                                                                                    */
  /**************************************************************************************/

  // FORMULATIONS

  // order of extrapolation of viscous term and convective term in pressure Neumann BC
  unsigned int order_extrapolation_pressure_nbc;

  // description: see enum declaration
  // The formulation of the convective term in the boundary conditions for the dual
  // splitting scheme (there are two occurrences, the g_u_hat term
  // arising from the divergence term on the right-hand side of the pressure Poisson
  // equation as well as the pressure NBC for the dual splitting scheme)can be chosen
  // independently from the type of formulation used for the discretization of the
  // convective term in the Navier-Stokes equations.
  // As a default parameter, FormulationConvectiveTerm::ConvectiveFormulation is
  // used (exploiting that div(u)=0 holds in the continuous case).
  FormulationConvectiveTerm formulation_convective_term_bc;

  // CONVECTIVE STEP

  // VISCOUS STEP

  // description: see enum declaration
  SolverViscous solver_viscous;

  // solver data for viscous step
  SolverData solver_data_viscous;

  // description: see enum declaration
  PreconditionerViscous preconditioner_viscous;

  // update preconditioner before solving the viscous step
  bool update_preconditioner_viscous;

  // Update preconditioner every ... time steps.
  // This variable is only used if update_preconditioner_viscous is true.
  unsigned int update_preconditioner_viscous_every_time_steps;

  // description: see declaration of MultigridData
  MultigridData multigrid_data_viscous;


  /**************************************************************************************/
  /*                                                                                    */
  /*                            PRESSURE-CORRECTION SCHEME                              */
  /*                                                                                    */
  /**************************************************************************************/

  // Newton solver data
  Newton::SolverData newton_solver_data_momentum;

  // description: see enum declaration
  SolverMomentum solver_momentum;

  // Solver data for (linearized) momentum equation
  SolverData solver_data_momentum;

  // description: see enum declaration
  MomentumPreconditioner preconditioner_momentum;

  // update preconditioner before solving the linear system of equations
  // only necessary if the operator changes during the simulation
  bool update_preconditioner_momentum;

  // Update preconditioner every ... Newton iterations (only relevant for
  // nonlinear problems, i.e., if the convective term is formulated implicitly)
  // This variable is only used if update_preconditioner_coupled = true.
  unsigned int update_preconditioner_momentum_every_newton_iter;

  // Update preconditioner every ... time steps.
  // This variable is only used if update_preconditioner_coupled = true.
  unsigned int update_preconditioner_momentum_every_time_steps;

  // description: see declaration of MultigridData
  MultigridData multigrid_data_momentum;

  // description: see enum declaration
  MultigridOperatorType multigrid_operator_type_momentum;

  // order of pressure extrapolation in case of incremental formulation
  // a value of 0 corresponds to non-incremental formulation
  // and a value >=1 to incremental formulation
  unsigned int order_pressure_extrapolation;

  // rotational formulation
  bool rotational_formulation;


  /**************************************************************************************/
  /*                                                                                    */
  /*                            COUPLED NAVIER-STOKES SOLVER                            */
  /*                                                                                    */
  /**************************************************************************************/

  // use a scaling of the continuity equation
  bool use_scaling_continuity;

  // scaling factor continuity equation
  double scaling_factor_continuity;

  // solver tolerances Newton solver
  Newton::SolverData newton_solver_data_coupled;

  // description: see enum declaration
  SolverCoupled solver_coupled;

  // Solver data for coupled solver
  SolverData solver_data_coupled;

  // description: see enum declaration
  PreconditionerCoupled preconditioner_coupled;

  // Update preconditioner
  bool update_preconditioner_coupled;

  // Update preconditioner every ... Newton iterations (only relevant for
  // nonlinear problems, i.e., if the convective term is formulated implicitly)
  // This variable is only used if update_preconditioner_coupled = true.
  unsigned int update_preconditioner_coupled_every_newton_iter;

  // Update preconditioner every ... time steps.
  // This variable is only used if update_preconditioner_coupled = true.
  unsigned int update_preconditioner_coupled_every_time_steps;

  // description: see enum declaration
  MomentumPreconditioner preconditioner_velocity_block;

  // description: see enum declaration
  MultigridOperatorType multigrid_operator_type_velocity_block;

  // description: see declaration
  MultigridData multigrid_data_velocity_block;

  // The momentum block is inverted "exactly" in block preconditioner
  // by solving the velocity convection-diffusion problem to a given
  // relative tolerance
  bool exact_inversion_of_velocity_block;

  // solver data for velocity block (only relevant if velocity block
  // is inverted exactly)
  SolverData solver_data_velocity_block;

  // description: see enum declaration
  SchurComplementPreconditioner preconditioner_pressure_block;

  // description: see declaration
  MultigridData multigrid_data_pressure_block;

  // The Laplace operator is inverted "exactly" in block preconditioner
  // by solving the Laplace problem to a given relative tolerance
  bool exact_inversion_of_laplace_operator;

  // solver data for Schur complement
  // (only relevant if exact_inversion_of_laplace_operator == true)
  SolverData solver_data_pressure_block;



  /**************************************************************************************/
  /*                                                                                    */
  /*                            SOLVE MASS SYSTEM (projection)                          */
  /*                                                                                    */
  /**************************************************************************************/
  // Used when matrix-free inverse mass operator is not avaliable, e.g in the case of HDIV.

  // solver data for solving mass system
  SolverData solver_data_mass;

  // description: see enum declaration
  PreconditionerMass preconditioner_mass;

  // Used when matrix-free inverse mass operator is not available and when the spatial
  // discretization is DG, e.g. simplex.
  bool solve_elementwise_mass_system_matrix_free;

  SolverData solver_data_elementwise_inverse_mass;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_ */
