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

using namespace dealii;

enum class ProblemType { Undefined, Steady, Unsteady};
enum class EquationType { Undefined, Stokes, NavierStokes };
enum class TreatmentOfConvectiveTerm { Undefined, Explicit, Implicit };

enum class TimeStepCalculation { Undefined, ConstTimeStepUserSpecified, ConstTimeStepCFL, AdaptiveTimeStepCFL };

enum class TemporalDiscretization { Undefined, BDFDualSplittingScheme, BDFCoupledSolution };
enum class SpatialDiscretization { DG, DGXWall};

enum class ProjectionType { Undefined, NoPenalty, DivergencePenalty, DivergenceAndContinuityPenalty };
enum class SolverProjection { LU, PCG };
enum class PreconditionerProjection { None, Jacobi, InverseMassMatrix };

enum class FormulationViscousTerm { DivergenceFormulation, LaplaceFormulation };
enum class InteriorPenaltyFormulationViscous { SIPG, NIPG };
enum class SolverViscous { PCG, GMRES };
enum class PreconditionerViscous { None, Jacobi, InverseMassMatrix, GeometricMultigrid };

enum class PreconditionerLinearizedNavierStokes { Undefined, None, BlockDiagonal, BlockTriangular, BlockTriangularFactorization };
enum class PreconditionerMomentum { Undefined, None, InverseMassMatrix, GeometricMultigrid };
enum class PreconditionerSchurComplement { Undefined, None, InverseMassMatrix, GeometricMultigrid, CahouetChabard };

class InputParameters
{
public:
  // standard constructor that initializes parameters
  InputParameters()
:
  problem_type(ProblemType::Undefined),
  equation_type(EquationType::Undefined),
  treatment_of_convective_term(TreatmentOfConvectiveTerm::Undefined),
  start_time(0.),
  end_time(-1.),
  max_number_of_steps(std::numeric_limits<unsigned int>::max()),
  calculation_of_time_step_size(TimeStepCalculation::Undefined),
  cfl(-1.),
  max_velocity(-1.),
  time_step_size(-1.),
  viscosity(-1.),
  temporal_discretization(TemporalDiscretization::Undefined),
  spatial_discretization(SpatialDiscretization::DG),
  order_time_integrator(1),
  start_with_low_order(true),
  use_symmetric_saddle_point_matrix(true),
  small_time_steps_stability(false),
  pure_dirichlet_bc(false),
  penalty_factor_divergence(1.),
  penalty_factor_continuity(1.),
  compute_divergence(false),
  divu_integrated_by_parts(false),
  divu_use_boundary_data(false),
  gradp_integrated_by_parts(false),
  gradp_use_boundary_data(false),
  IP_factor_pressure(1.),
  IP_factor_viscous(1.),
  abs_tol_newton(1.e-20),
  rel_tol_newton(1.e-12),
  max_iter_newton(std::numeric_limits<unsigned int>::max()),
  abs_tol_linear(1.e-20),
  rel_tol_linear(1.e-12),
  max_iter_linear(std::numeric_limits<unsigned int>::max()),
  abs_tol_pressure(1.e-20),
  rel_tol_pressure(1.e-12),
  abs_tol_projection(1.e-20),
  rel_tol_projection(1.e-12),
  abs_tol_viscous(1.e-20),
  rel_tol_viscous(1.e-12),
  solver_poisson(SolverPoisson::PCG),
  preconditioner_poisson(PreconditionerPoisson::GeometricMultigrid),
  multigrid_smoother(MultigridSmoother::Chebyshev),
  multigrid_coarse_grid_solver(MultigridCoarseGridSolver::coarse_chebyshev_smoother),
  projection_type(ProjectionType::Undefined),
  solver_projection(SolverProjection::PCG),
  preconditioner_projection(PreconditionerProjection::InverseMassMatrix),
  formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
  IP_formulation_viscous(InteriorPenaltyFormulationViscous::SIPG),
  solver_viscous(SolverViscous::PCG),
  preconditioner_viscous(PreconditionerViscous::InverseMassMatrix),
  multigrid_smoother_viscous(MultigridSmoother::Chebyshev),
  multigrid_coarse_grid_solver_viscous(MultigridCoarseGridSolver::coarse_chebyshev_smoother),
  preconditioner_linearized_navier_stokes(PreconditionerLinearizedNavierStokes::Undefined),
  preconditioner_momentum(PreconditionerMomentum::Undefined),
  preconditioner_schur_complement(PreconditionerSchurComplement::Undefined),
  output_solver_info_every_timesteps(1),
  output_start_time(std::numeric_limits<double>::max()),
  output_interval_time(std::numeric_limits<double>::max()),
  restart_interval_time(std::numeric_limits<double>::max()),
  restart_interval_wall_time(std::numeric_limits<double>::max()),
  restart_interval_step(std::numeric_limits<unsigned int>::max()),
  output_prefix("indexa"),
  error_calc_start_time(std::numeric_limits<double>::max()),
  error_calc_interval_time(std::numeric_limits<double>::max()),
  analytical_solution_available(false),
  statistics_start_time(std::numeric_limits<double>::max()),
  statistics_every(1),
  cs(0.),
  ml(0.),
  variabletauw(true),
  dtauw(1.),
  max_wdist_xwall(-1.)
  {
  };

  void set_input_parameters();

  void check_parameters()
  {
    AssertThrow(problem_type != ProblemType::Undefined,ExcMessage("parameter must be defined"));
    AssertThrow(equation_type != EquationType::Undefined,ExcMessage("parameter must be defined"));
    AssertThrow(treatment_of_convective_term != TreatmentOfConvectiveTerm::Undefined,ExcMessage("parameter must be defined"));
    AssertThrow(end_time > start_time,ExcMessage("parameter end_time must be defined"));
    AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,ExcMessage("parameter must be defined"));
    if(calculation_of_time_step_size != TimeStepCalculation::ConstTimeStepUserSpecified)
    {
      AssertThrow(cfl > 0.,ExcMessage("parameter must be defined"));
      AssertThrow(max_velocity > 0.,ExcMessage("parameter must be defined"));
    }
    if(calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
      AssertThrow(time_step_size > 0.,ExcMessage("parameter must be defined"));
    AssertThrow(viscosity > 0.,ExcMessage("parameter must be defined"));
    AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,ExcMessage("parameter must be defined"));

    if(temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      AssertThrow(projection_type !=ProjectionType::Undefined,ExcMessage("parameter must be defined"));
    }

    if(temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      AssertThrow(preconditioner_linearized_navier_stokes != PreconditionerLinearizedNavierStokes::Undefined,ExcMessage("parameter must be defined"));
      AssertThrow(preconditioner_momentum != PreconditionerMomentum::Undefined,ExcMessage("parameter must be defined"));
      AssertThrow(preconditioner_schur_complement != PreconditionerSchurComplement::Undefined,ExcMessage("parameter must be defined"));
    }
  }

  // describes whether a steady state problem or unsteady problem is solved
  ProblemType problem_type;

  // describes the physical/mathematical model that has to be solved, i.e. Stokes vs. NavierStokes
  EquationType equation_type;

  // the convective term can be either treated explicitly or implicitly
  TreatmentOfConvectiveTerm treatment_of_convective_term;

  // start and end time of time interval
  double start_time;
  double end_time;

  // maximum number of time steps
  unsigned int max_number_of_steps;

  // calculation of time step size
  TimeStepCalculation  calculation_of_time_step_size;

  // cfl number: note that this cfl number is the first in a series of cfl numbers when performing temporal convergence tests,
  // i.e., cfl_real = cfl, cfl/2, cfl/4, cfl/8, ...
  double cfl;

  // maximum velocity needed when calculating the time step size according to cfl condition
  double max_velocity;

  // user specified time step size: note that this time_step_size is the first in a series of time_step_size's when
  // performing temporal convergence tests, i.e., time_step_size_real = time_step_size, time_step_size/2, ...
  double time_step_size;

  // kinematic viscosity nu
  double viscosity;

  // temporal discretization method
  TemporalDiscretization temporal_discretization;

  // spatial discretization method
  SpatialDiscretization spatial_discretization;

  // order of BDF time integration scheme and extrapolation scheme
  unsigned int order_time_integrator;
  // start time integrator with low order time integrator, i.e. first order Euler method
  bool start_with_low_order;

  // use symmetric saddle point matrix for coupled solver:
  // continuity equation formulated as: - div(u) = 0 -> symmetric formulation
  //                                      div(u) = 0 -> non-symmetric formulation
  bool use_symmetric_saddle_point_matrix;

  // use small time steps stability approach (similar to approach of Leriche et al.)
  bool small_time_steps_stability;

  // special case of pure Dirichlet BCs on whole boundary
  bool pure_dirichlet_bc;

  // penalty factor of divergence penalty term in projection step
  double penalty_factor_divergence;
  // penalty factor of divergence penalty term in projection step
  double penalty_factor_continuity;

  // compute divergence of intermediate velocity field to verify divergence penalty method
  bool compute_divergence;

  // integration by parts of div(U_tilde) on rhs of pressure Poisson equation
  bool divu_integrated_by_parts;
  // use boundary data if integrated by parts
  bool divu_use_boundary_data;
  // integration by parts of grad(P) on rhs projection step
  bool gradp_integrated_by_parts;
  // use boundary data if integrated by parts
  bool gradp_use_boundary_data;

  // interior penalty parameter scaling factor for pressure Poisson equation
  double IP_factor_pressure;
  // interior penalty parameter scaling factor for Helmholtz equation of viscous step
  double IP_factor_viscous;

  // solver tolerances Newton solver
  double abs_tol_newton;
  double rel_tol_newton;
  unsigned int max_iter_newton;

  // solver tolerances for linearized problem of Newton solver
  double abs_tol_linear;
  double rel_tol_linear;
  unsigned int max_iter_linear;

  // solver tolerances for pressure Poisson equation
  double abs_tol_pressure;
  double rel_tol_pressure;

  // solver tolerances for projection step
  double abs_tol_projection;
  double rel_tol_projection;

  // solver tolerances for Helmholtz equation of viscous step
  double abs_tol_viscous;
  double rel_tol_viscous;

  // type of pressure Poisson solver
  SolverPoisson solver_poisson;
  // preconditioner type for solution of pressure Poisson equation
  PreconditionerPoisson preconditioner_poisson;
  // multigrid smoother pressure Poisson equation
  MultigridSmoother multigrid_smoother;
  // multigrid coarse grid solver pressure Poisson equation
  MultigridCoarseGridSolver multigrid_coarse_grid_solver;

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
  // multigrid smoother pressure Poisson equation
  MultigridSmoother multigrid_smoother_viscous;
  // multigrid coarse grid solver pressure Poisson equation
  MultigridCoarseGridSolver multigrid_coarse_grid_solver_viscous;

  // preconditioner linearized Navier-Stokes problem
  PreconditionerLinearizedNavierStokes preconditioner_linearized_navier_stokes;
  // preconditioner for (1,1) velocity/momentum block in case of block preconditioning
  PreconditionerMomentum preconditioner_momentum;
  // preconditioner for (2,2) pressure/Schur complement block in case of block preconditioning
  PreconditionerSchurComplement preconditioner_schur_complement;

  // show solver performance (wall time, number of iterations) every ... timesteps
  unsigned int output_solver_info_every_timesteps;

  // before then no output will be written
  double output_start_time;
  // specifies the time interval in which output is written
  double output_interval_time;

  // specifies the time interval in which restarts are written, starting from start_time
  double restart_interval_time;
  // specifies the wall time interwal in which restarts are written
  double restart_interval_wall_time;
  // specifies the restart interval via number of time steps
  unsigned int restart_interval_step;

  // name of generated output files
  std::string output_prefix;

  // before then no error calculation will be performed
  double error_calc_start_time;
  // specifies the time interval in which error calculation is performed
  double error_calc_interval_time;

  // to calculate the error an analytical solution to the problem has to be available
  bool analytical_solution_available;

  // before then no statistics calculation will be performed
  double statistics_start_time;
  // calculate statistics every "statistics_every" time steps
  unsigned int statistics_every;

  // Smagorinsky constant
  double cs;
  // mixing-length model for xwall
  double ml;
  // xwall with adaptive wall shear stress
  bool variabletauw;
  // delta tauw if adaptive between 0 and 1
  double dtauw;
  // max wall distance of enriched elements
  double max_wdist_xwall;

};

#endif /* INCLUDE_INPUTPARAMETERS_H_ */
