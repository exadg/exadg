/*
 * input_parameters.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_

#include "deal.II/base/conditional_ostream.h"

#include "../../functionalities/restart_data.h"
#include "../../functionalities/solver_info_data.h"

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/newton/newton_solver_data.h"
#include "../../solvers_and_preconditioners/preconditioner/enum_types.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"
#include "../../time_integration/enum_types.h"

#include "enum_types.h"

namespace IncNS
{
class InputParameters
{
public:
  // standard constructor that initializes parameters
  InputParameters();

  void
  check_input_parameters();

  bool
  convective_problem() const;

  bool
  viscous_problem() const;

  bool
  nonlinear_problem_has_to_be_solved() const;

  bool
  linear_problem_has_to_be_solved() const;

  unsigned int
  get_degree_p() const;

  void
  print(ConditionalOStream & pcout, std::string const & name);

private:
  void
  print_parameters_mathematical_model(ConditionalOStream & pcout);

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout);

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout);

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout);

  void
  print_parameters_turbulence(ConditionalOStream & pcout);

  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout);

  void
  print_parameters_pressure_poisson(ConditionalOStream & pcout);

  void
  print_parameters_projection_step(ConditionalOStream & pcout);

  void
  print_parameters_dual_splitting(ConditionalOStream & pcout);

  void
  print_parameters_pressure_correction(ConditionalOStream & pcout);

  void
  print_parameters_coupled_solver(ConditionalOStream & pcout);

public:
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // number of space dimensions
  unsigned int dim;

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

  /**************************************************************************************/
  /*                                                                                    */
  /*                 Arbitrary Lagrangian-Eulerian formulation (ALE)                    */
  /*                                                                                    */
  /**************************************************************************************/

  // use true to activate ALE formulation, otherwise the standard Eulerian formulation
  // with fixed mesh will be used
  bool ale_formulation;

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

  // Boussinesg term
  double               thermal_expansion_coefficient;
  Tensor<1, 3, double> gravitational_force;

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
  TimeIntegratorOIF time_integrator_oif;

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

  // cfl number for operator-integration-factor splitting (has to be smaller than the
  // critical time step size arising from the CFL restriction)
  double cfl_oif;

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

  // order of BDF time integration scheme and extrapolation scheme
  unsigned int order_time_integrator;

  // start time integrator with low order time integrator, i.e., first order Euler method
  bool start_with_low_order;

  // number of refinement steps for time step size
  //   default: use dt_refinements = 0
  unsigned int dt_refinements;

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

  // triangulation type
  TriangulationType triangulation_type;

  // Polynomial degree of velocity shape functions
  unsigned int degree_u;

  // Polynomial degree used for pressure shape functions
  DegreePressure degree_p;

  // Type of mapping (polynomial degree) use for geometry approximation
  MappingType mapping;

  // Number of mesh refinement steps
  unsigned int h_refinements;

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

  // special case of pure Dirichlet BCs on whole boundary
  bool pure_dirichlet_bc;

  // adjust pressure level in case of pure Dirichlet BC's where
  // the pressure is only defined up to an additive constant
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
  /*                                     TURBULENCE                                     */
  /*                                                                                    */
  /**************************************************************************************/

  // use turbulence model
  bool use_turbulence_model;

  // scaling factor for turbulent viscosity model
  double turbulence_model_constant;

  // turbulence model
  TurbulenceEddyViscosityModel turbulence_model;


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

  // FORMULATIONS

  // For projection methods, boundary conditions have to be accessed at previous times
  // during a time step. For the dual splitting scheme, the acceleration is required in
  // the pressure Neumann boundary condition. Since these values are not available
  // analytically in a general setting, there is the possibility to (fill and) store
  // these boundary conditions in vectors and access them later when needed (and compute
  // the time derivative numerically to obtain the acceleration).
  // Hence, this parameter should be set to true, which is also the default value of this
  // parameter.
  // Choosing false means that the user has to make sure that the boundary conditions
  // at previous times are accessible and evaluated correctly by just setting the parameter
  // time of the Function<dim>. Furthermore, the user has to provide the time derivative
  // of the velocity (acceleration) analytically in case of the dual splitting scheme.
  bool store_previous_boundary_values;

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
  // splitting scheme can be chosen independently from the type of formulation used for
  // the discretization of the convective term in the Navier-Stokes equations.
  // As a default parameter, FormulationConvectiveTerm::ConvectiveFormulation should be
  // used (exploiting that div(u)=0 holds in the continuous case). The background is
  // that for the BDF3 time integration scheme, instabilities have been observed when
  // using FormulationConvectiveTerm::DivergenceFormulation in the boundary conditions
  // of the dual splitting scheme (where there are two occurrences, the g_u_hat term
  // arising form the divergence term on the right-hand side of the pressure Poisson
  // equation as well as the pressure NBC for the dual splitting scheme).
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
  NewtonSolverData newton_solver_data_momentum;

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

  // use symmetric saddle point matrix for coupled solver:
  // continuity equation formulated as: - div(u) = 0 -> symmetric formulation
  //                                      div(u) = 0 -> non-symmetric formulation
  //  bool use_symmetric_saddle_point_matrix;

  // use a scaling of the continuity equation
  bool use_scaling_continuity;

  // scaling factor continuity equation
  double scaling_factor_continuity;

  // solver tolerances Newton solver
  NewtonSolverData newton_solver_data_coupled;

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

  // description: see enum declaration
  DiscretizationOfLaplacian discretization_of_laplacian;

  // description: see declaration
  MultigridData multigrid_data_pressure_block;

  // The Laplace operator is inverted "exactly" in block preconditioner
  // by solving the Laplace problem to a given relative tolerance
  bool exact_inversion_of_laplace_operator;

  // solver data for schur complement
  // (only relevant if exact_inversion_of_laplace_operator == true)
  SolverData solver_data_pressure_block;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_ */
