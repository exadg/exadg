/*
 * InputParameters.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INPUTPARAMETERS_H_
#define INCLUDE_INPUTPARAMETERS_H_

#include "PoissonSolverInputParameters.h"
#include "MultigridInputParameters.h"

enum class ProblemType { Steady, Unsteady };
enum class EquationType { Stokes, NavierStokes };
enum class TreatmentOfConvectiveTerm { Explicit, Implicit };

enum class TimeStepCalculation { ConstTimeStepUserSpecified, ConstTimeStepCFL, AdaptiveTimeStepCFL };

enum class TemporalDiscretization { BDFDualSplittingScheme, BDFCoupledSolution };

enum class ProjectionType { NoPenalty, DivergencePenalty, DivergenceAndContinuityPenalty };
enum class SolverProjection { LU, PCG };
enum class PreconditionerProjection { None, Jacobi, InverseMassMatrix };

enum class FormulationViscousTerm { DivergenceFormulation, LaplaceFormulation };
enum class InteriorPenaltyFormulationViscous { SIPG, NIPG };
enum class SolverViscous { PCG, GMRES };
enum class PreconditionerViscous { None, Jacobi, InverseMassMatrix, GeometricMultigrid };

enum class PreconditionerLinearizedNavierStokes { None, BlockDiagonal, BlockTriangular };
enum class PreconditionerMomentum { None, InverseMassMatrix, GeometricMultigrid };
enum class PreconditionerSchurComplement {None, InverseMassMatrix, GeometricMultigrid };

class InputParameters
{
public:
  // standard constructor that initializes parameters
  InputParameters();

  // describes whether a steady state problem or unsteady problem is solved
  ProblemType const problem_type;

  // describes the physical/mathematical model that has to be solved, i.e. Stokes vs. NavierStokes
  EquationType const equation_type;

  // the convective term can be either treated explicitly or implicitly
  TreatmentOfConvectiveTerm const treatment_of_convective_term;

  // start and end time of time interval
  double const start_time;
  double const end_time;

  // maximum number of time steps
  unsigned int const max_number_of_steps;

  // calculation of time step size
  TimeStepCalculation const  calculation_of_time_step_size;

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

  // temporal discretization method
  TemporalDiscretization temporal_discretization;

  // order of BDF time integration scheme and extrapolation scheme
  unsigned int const order_time_integrator;
  // start time integrator with low order time integrator, i.e. first order Euler method
  bool const start_with_low_order;

  // use symmetric saddle point matrix for coupled solver:
  // continuity equation formulated as: - div(u) = 0 -> symmetric formulation
  //                                      div(u) = 0 -> non-symmetric formulation
  bool const use_symmetric_saddle_point_matrix;

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
  // use boundary data if integrated by parts
  bool const divu_use_boundary_data;
  // integration by parts of grad(P) on rhs projection step
  bool const gradp_integrated_by_parts;
  // use boundary data if integrated by parts
  bool const gradp_use_boundary_data;

  // interior penalty parameter scaling factor for pressure Poisson equation
  double const IP_factor_pressure;
  // interior penalty parameter scaling factor for Helmholtz equation of viscous step
  double const IP_factor_viscous;

  // solver tolerances Newton solver
  double const abs_tol_newton;
  double const rel_tol_newton;
  unsigned int const max_iter_newton;

  // solver tolerances for linearized problem of Newton solver
  double const abs_tol_linear;
  double const rel_tol_linear;
  unsigned int const max_iter_linear;

  // solver tolerances for pressure Poisson equation
  double const abs_tol_pressure;
  double const rel_tol_pressure;

  // solver tolerances for projection step
  double const abs_tol_projection;
  double const rel_tol_projection;

  // solver tolerances for Helmholtz equation of viscous step
  double const abs_tol_viscous;
  double const rel_tol_viscous;

  // type of pressure Poisson solver
  SolverPoisson const solver_poisson;
  // preconditioner type for solution of pressure Poisson equation
  PreconditionerPoisson const preconditioner_poisson;
  // multigrid smoother pressure Poisson equation
  MultigridSmoother const multigrid_smoother;
  // multigrid coarse grid solver pressure Poisson equation
  MultigridCoarseGridSolver const multigrid_coarse_grid_solver;

  // projection type: standard projection (no penalty term), divergence penalty term, divergence and continuity penalty term (weak projection)
  ProjectionType const projection_type;
  // type of projections solver
  SolverProjection const solver_projection;
  // preconditioner type for solution of projection step
  PreconditionerProjection const preconditioner_projection;

  // formulation of viscous term: divergence formulation or Laplace formulation
  FormulationViscousTerm const formulation_viscous_term;
  // interior penalty formulation of viscous term: SIPG (symmetric IP) or NIPG (non-symmetric IP)
  InteriorPenaltyFormulationViscous const IP_formulation_viscous;
  // Solver type for solution of viscous step
  SolverViscous const solver_viscous;
  // Preconditioner type for solution of viscous step
  PreconditionerViscous const preconditioner_viscous;
  // multigrid smoother pressure Poisson equation
  MultigridSmoother const multigrid_smoother_viscous;
  // multigrid coarse grid solver pressure Poisson equation
  MultigridCoarseGridSolver const multigrid_coarse_grid_solver_viscous;

  // preconditioner linearized Navier-Stokes problem
  PreconditionerLinearizedNavierStokes const preconditioner_linearized_navier_stokes;
  // preconditioner for (1,1) velocity/momentum block in case of block preconditioning
  PreconditionerMomentum const preconditioner_momentum;
  // preconditioner for (2,2) pressure/Schur complement block in case of block preconditioning
  PreconditionerSchurComplement const preconditioner_schur_complement;

  // show solver performance (wall time, number of iterations) every ... timesteps
  unsigned int const output_solver_info_every_timesteps;

  // before then no output will be written
  double const output_start_time;
  // specifies the time interval in which output is written
  double const output_interval_time;

  // specifies the time interval in which restarts are written, starting from start_time
  double const restart_interval_time;
  // specifies the wall time interwal in which restarts are written
  double const restart_interval_wall_time;
  // specifies the restart interval via number of time steps
  unsigned int const restart_interval_step;

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
