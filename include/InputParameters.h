/*
 * InputParameters.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INPUTPARAMETERS_H_
#define INCLUDE_INPUTPARAMETERS_H_

enum class TimeStepCalculation { ConstTimeStepUserSpecified, ConstTimeStepCFL, AdaptiveTimeStepCFL };

enum class ProjectionType { NoPenalty, DivergencePenalty, DivergenceAndContinuityPenalty };
enum class SolverProjection { LU, PCG };
enum class PreconditionerProjection { None, Jacobi, InverseMassMatrix };

enum class FormulationViscousTerm { DivergenceFormulation, LaplaceFormulation };
enum class InteriorPenaltyFormulationViscous { SIPG, NIPG };
enum class SolverViscous { PCG, GMRES };
enum class PreconditionerViscous { None, Jacobi, InverseMassMatrix };

class InputParameters
{
public:
  // standard constructor that initializes parameters
  InputParameters();

  // start and end time of time interval
  double const start_time;
  double const end_time;

  // maximum number of time steps
  unsigned int const max_number_of_steps;

  // calculation of time step size
  TimeStepCalculation calculation_of_time_step_size;

  // cfl number: note that this cfl number is the first in a series of cfl numbers when performing temporal convergence tests,
  // i.e., cfl_real = cfl, cfl/2, cfl/4, cfl/8, ...
  double const cfl;

  // maximum velocity needed when calculating the time step size according to cfl condition
  double const max_velocity;

  // user specified time step size: note that this time_step_size is the first in a series of time_step_size's when
  // performing temporal convergence tests, i.e., time_step_size_real = time_step_size, time_step_size/2, ...
  double const time_step_size;

  // kinematic viscosity nu
  double const viscosity;

  // order of BDF time integration scheme and extrapolation scheme
  unsigned int const order_time_integrator;
  // start time integrator with low order time integrator, i.e. first order Euler method
  bool const start_with_low_order;

  // solve unsteady Stokes equations
  bool const solve_stokes_equations;

  // solve convective step implicitly
  bool const convective_step_implicit;

  // use small time steps stability approach (similar to approach of Leriche et al.)
  bool const small_time_steps_stability;

  // special case of pure Dirichlet BCs on whole boundary
  bool const pure_dirichlet_bc;

  // penalty factor of divergence penalty term in projection step
  double const penalty_factor_divergence;
  // penalty factor of divergence penalty term in projection step
  double const penalty_factor_continuity;

  // compute divergence of intermediate velocity field to verify divergence penalty method
  bool const compute_divergence;

  // integration by parts of div(U_tilde) on rhs of pressure Poisson equation
  bool const divu_integrated_by_parts;
  // integration by parts of grad(P) on rhs projection step
  bool const gradp_integrated_by_parts;

  // interior penalty parameter scaling factor for pressure Poisson equation
  double const IP_factor_pressure;
  // interior penalty parameter scaling factor for Helmholtz equation of viscous step
  double const IP_factor_viscous;

  // solver tolerances for pressure Poisson equation
  double const abs_tol_pressure;
  double const rel_tol_pressure;

  // solver tolerances for projection step
  double const abs_tol_projection;
  double const rel_tol_projection;

  // solver tolerances for Helmholtz equation of viscous step
  double const abs_tol_viscous;
  double const rel_tol_viscous;

  // projection type: standard projection (no penalty term), divergence penalty term, divergence and continuity penalty term (weak projection)
  ProjectionType projection_type;
  // type of projections solver
  SolverProjection solver_projection;
  // preconditioner type for solution of projection step
  PreconditionerProjection preconditioner_projection;

  // formulation of viscous term: divergence formulation or Laplace formulation
  FormulationViscousTerm formulation_viscous_term;
  // interior penalty formulation of viscous term: SIPG (symmetric IP) or NIPG (non-symmetric IP)
  InteriorPenaltyFormulationViscous IP_formulation_viscous;
  // Solver type for solution of viscous step
  SolverViscous solver_viscous;
  // Preconditioner type for solution of viscous step
  PreconditionerViscous preconditioner_viscous;

  // show solver performance (wall time, number of iterations) every ... timesteps
  unsigned int const output_solver_info_every_timesteps;

  // before then no output will be written
  double const output_start_time;
  // specifies the time interval in which output is written
  double const output_interval_time;

  // name of generated output files
  std::string const output_prefix;

  // before then no error calculation will be performed
  double const error_calc_start_time;
  // specifies the time interval in which error calculation is performed
  double const error_calc_interval_time;

  // to calculate the error an analytical solution to the problem has to be available
  bool const analytical_solution_available;

  // before then no statistics calculation will be performed
  double const statistics_start_time;
  // calculate statistics every "statistics_every" time steps
  unsigned int const statistics_every;

  // Smagorinsky constant
  double const cs;
  // mixing-length model for xwall
  double const ml;
  // xwall with adaptive wall shear stress
  bool const variabletauw;
  // delta tauw if adaptive between 0 and 1
  double const dtauw;
};

#endif /* INCLUDE_INPUTPARAMETERS_H_ */
