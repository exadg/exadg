/*
 * input_parameters.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_

#include "deal.II/base/conditional_ostream.h"

#include "../../functionalities/print_functions.h"
#include "../../functionalities/restart_data.h"
#include "../../incompressible_navier_stokes/postprocessor/kinetic_energy_data.h"
#include "../../incompressible_navier_stokes/postprocessor/kinetic_energy_spectrum_data.h"
#include "../../incompressible_navier_stokes/postprocessor/lift_and_drag_data.h"
#include "../../incompressible_navier_stokes/postprocessor/output_data_navier_stokes.h"
#include "../../incompressible_navier_stokes/postprocessor/pressure_difference_data.h"
#include "../../incompressible_navier_stokes/postprocessor/turbulence_statistics_data.h"
#include "../../incompressible_navier_stokes/postprocessor/turbulent_channel_data.h"
#include "../../postprocessor/error_calculation_data.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/newton/newton_solver_data.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"
#include "../postprocessor/line_plot_data.h"
#include "../postprocessor/mean_velocity_calculator.h"

namespace IncNS
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  ProblemType refers to the underlying physics of the flow problem and describes
 *  whether the considered flow problem is expected to be a steady or unsteady solution
 *  of the incompressible Navier-Stokes equations. This essentially depends on the
 *  Reynolds number but, for example, also on the boundary conditions (in case of
 *  time dependent boundary conditions, the problem type is always unsteady).
 */
enum class ProblemType
{
  Undefined,
  Steady,
  Unsteady
};

/*
 *  EquationType describes the physical/mathematical model that has to be solved,
 *  i.e., Stokes equations or Navier-Stokes equations
 */
enum class EquationType
{
  Undefined,
  Stokes,
  NavierStokes
};

/*
 *  Formulation of viscous term: divergence formulation or Laplace formulation
 */
enum class FormulationViscousTerm
{
  Undefined,
  DivergenceFormulation,
  LaplaceFormulation
};

/*
 *  Formulation of convective term: divergence formulation or convective formulation
 */
enum class FormulationConvectiveTerm
{
  Undefined,
  DivergenceFormulation,
  ConvectiveFormulation,
  EnergyPreservingFormulation
};

/**************************************************************************************/
/*                                                                                    */
/*                                 PHYSICAL QUANTITIES                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                             TEMPORAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  SolverType refers to the numerical solution of the incompressible Navier-Stokes
 *  equations and describes whether a steady or an unsteady solver is used.
 *  While it does not make sense to solve an unsteady problem with a steady solver,
 *  a steady problem can be solved (potentially more efficiently) by using an
 *  unsteady solver.
 */
enum class SolverType
{
  Undefined,
  Steady,
  Unsteady
};

/*
 *  Temporal discretization method
 */
enum class TemporalDiscretization
{
  Undefined,
  BDFDualSplittingScheme,
  BDFPressureCorrection,
  BDFCoupledSolution
};

/*
 *  the convective term can be either treated explicitly or implicitly
 */
enum class TreatmentOfConvectiveTerm
{
  Undefined,
  Explicit,
  ExplicitOIF,
  Implicit
};

/*
 *  Temporal discretization method for OIF splitting:
 *
 *    Explicit Runge-Kutta methods
 */
enum class TimeIntegratorOIF
{
  Undefined,
  ExplRK1Stage1,
  ExplRK2Stage2,
  ExplRK3Stage3,
  ExplRK4Stage4,
  ExplRK3Stage4Reg2C,
  ExplRK3Stage7Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg2C,
  ExplRK4Stage8Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg3C,
  ExplRK5Stage9Reg2S
};


/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  ConstTimeStepUserSpecified,
  ConstTimeStepCFL,
  AdaptiveTimeStepCFL,
  ConstTimeStepMaxEfficiency // only relevant for analytical test cases with optimal rates of
                             // convergence in space
};

/*
 *  Pseudo-timestepping for steady-state problems:
 *  Define convergence criterion that is used to terminate simulation
 *
 *  option ResidualSteadyNavierStokes:
 *   - evaluate residual of steady, coupled incompressible Navier-Stokes equations
 *     and terminate simulation if norm of residual fulfills tolerances
 *   - can be used for the coupled solution approach
 *   - can be used for the pressure-correction scheme in case the incremental
 *     formulation is used (for the nonincremental formulation the steady-state
 *     solution cannot fulfill the residual of the steady Navier-Stokes equations
 *     in general due to the splitting error)
 *   - cannot be used for the dual splitting scheme (due to the splitting error
 *     the residual of the steady Navier-Stokes equations is not fulfilled)
 *
 *  option SolutionIncrement:
 *   - calculate solution increment from one time step to the next and terminate
 *     simulation if solution doesn't change any more (defined by tolerances)
 */
enum class ConvergenceCriterionSteadyProblem
{
  Undefined,
  ResidualSteadyNavierStokes,
  SolutionIncrement
};


/**************************************************************************************/
/*                                                                                    */
/*                              SPATIAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Type of imposition of Dirichlet BC's:
 *
 *  direct: u⁺ = g
 *  mirror: u⁺ = -u⁻ + 2g
 *
 *  A direct imposition might be advantageous with respect to the CFL condition
 *  possibly allowing to use significantly larger time step sizes.
 */
enum class TypeDirichletBCs
{
  Direct,
  Mirror
};

/*
 *  Interior penalty formulation of viscous term:
 *  SIPG (symmetric IP) or NIPG (non-symmetric IP)
 */
enum class InteriorPenaltyFormulation
{
  Undefined,
  SIPG,
  NIPG
};


/*
 *  Penalty term in case of divergence formulation:
 *  not symmetrized: penalty term identical to Laplace formulation, tau * [[u]]
 *  symmetrized: penalty term = tau * ([[u]] + [[u]]^T)
 */
enum class PenaltyTermDivergenceFormulation
{
  Undefined,
  Symmetrized,
  NotSymmetrized
};

enum class AdjustPressureLevel
{
  ApplyZeroMeanValue,
  ApplyAnalyticalMeanValue,
  ApplyAnalyticalSolutionInPoint
};

enum class ContinuityPenaltyComponents
{
  Undefined,
  All,
  Normal
};

enum class TypePenaltyParameter
{
  Undefined,
  ConvectiveTerm,
  ViscousTerm,
  ViscousAndConvectiveTerms
};

/**************************************************************************************/
/*                                                                                    */
/*                              NUMERICAL PARAMETERS                                  */
/*                                                                                    */
/**************************************************************************************/


/*
 * Elementwise preconditioner for block Jacobi preconditioner (only relevant for
 * elementwise iterative solution procedure)
 */
enum class PreconditionerBlockDiagonal
{
  Undefined,
  None,
  InverseMassMatrix
};


/**************************************************************************************/
/*                                                                                    */
/*                        HIGH-ORDER DUAL SPLITTING SCHEME                            */
/*                                                                                    */
/**************************************************************************************/


/*
 *  Solver for pressure Poisson equation
 */
enum class SolverPressurePoisson
{
  PCG,
  FGMRES
};

/*
 *  Preconditioner type for solution of pressure Poisson equation
 */
enum class PreconditionerPressurePoisson
{
  None,
  Jacobi,
  GeometricMultigrid
};

/*
 *  projection type: standard projection (no penalty term),
 *  divergence penalty term, divergence and continuity penalty term (weak projection)
 */
enum class ProjectionType
{
  Undefined,
  NoPenalty,
  DivergencePenalty,
  DivergenceAndContinuityPenalty
};

/*
 *  Type of projection solver
 */
enum class SolverProjection
{
  LU,
  PCG
};

/*
 *  Preconditioner type for solution of projection step
 */
enum class PreconditionerProjection
{
  None,
  PointJacobi,
  BlockJacobi,
  InverseMassMatrix
};

/*
 *  Solver type for solution of viscous step
 */
enum class SolverViscous
{
  PCG,
  GMRES,
  FGMRES
};

/*
 *  Preconditioner type for solution of viscous step
 */
enum class PreconditionerViscous
{
  None,
  InverseMassMatrix,
  PointJacobi,
  BlockJacobi,
  GeometricMultigrid
};

/**************************************************************************************/
/*                                                                                    */
/*                             PRESSURE-CORRECTION SCHEME                             */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Solver type for solution of momentum equation
 */
enum class SolverMomentum
{
  PCG,
  GMRES,
  FGMRES
};

/*
 *  Preconditioner type for solution of momentum equation:
 *
 *  see coupled solution approach below
 */


/**************************************************************************************/
/*                                                                                    */
/*                            COUPLED NAVIER-STOKES SOLVER                            */
/*                                                                                    */
/**************************************************************************************/

/*
 *   Solver for linearized Navier-Stokes problem
 */
enum class SolverLinearizedNavierStokes
{
  Undefined,
  GMRES,
  FGMRES
};

/*
 *  Preconditioner type for linearized Navier-Stokes problem
 */
enum class PreconditionerLinearizedNavierStokes
{
  Undefined,
  None,
  BlockDiagonal,
  BlockTriangular,
  BlockTriangularFactorization
};

/*
 *  preconditioner for (1,1) velocity/momentum block in case of block preconditioning
 */
enum class MomentumPreconditioner
{
  Undefined,
  None,
  PointJacobi,
  BlockJacobi,
  InverseMassMatrix,
  VelocityDiffusion,
  VelocityConvectionDiffusion
};


/*
 *  Preconditioner for (2,2) pressure/Schur complement block in case of block preconditioning
 */
enum class SchurComplementPreconditioner
{
  Undefined,
  None,
  InverseMassMatrix,
  LaplaceOperator,
  CahouetChabard,
  Elman,
  PressureConvectionDiffusion
};


/*
 *  Discretization of Laplacian: B: negative divergence operator, B^T gradient operator
 *  classical (BB^T is approximated by negative Laplace operator),
 *  compatible (BM^{-1}B^T)
 */
enum class DiscretizationOfLaplacian
{
  Undefined,
  Classical,
  Compatible
};



/**************************************************************************************/
/*                                                                                    */
/*                                     TURBULENCE                                     */
/*                                                                                    */
/**************************************************************************************/

/*
 * Set the turbulence modeling approach for xwall
 */
enum class XWallTurbulenceApproach
{
  Undefined,
  None,
  RANSSpalartAllmaras,
  ClassicalDESSpalartAllmaras,
  MultiscaleDESSpalartAllmaras
};

/*
 *  Algebraic subgrid-scale turbulence models for LES
 *
 *  Standard constants according to literature:
 *    Smagorinsky: 0.165
 *    Vreman: 0.28
 *    WALE: 0.50
 *    Sigma: 1.35
 */
enum class TurbulenceEddyViscosityModel
{
  Undefined,
  Smagorinsky,
  Vreman,
  WALE,
  Sigma
};



/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section

// mass conservation data

struct MassConservationData
{
  MassConservationData()
    : calculate_error(false),
      start_time(std::numeric_limits<double>::max()),
      sample_every_time_steps(std::numeric_limits<unsigned int>::max()),
      filename_prefix("indexa"),
      reference_length_scale(1.0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate_error == true)
    {
      pcout << "  Analysis of divergence and mass error:" << std::endl;
      print_parameter(pcout, "Calculate error", calculate_error);
      print_parameter(pcout, "Start time", start_time);
      print_parameter(pcout, "Sample every timesteps", sample_every_time_steps);
      print_parameter(pcout, "Filename prefix", filename_prefix);
    }
  }

  bool         calculate_error;
  double       start_time;
  unsigned int sample_every_time_steps;
  std::string  filename_prefix;
  double       reference_length_scale;
};


/*
 * inflow data: use velocity at the outflow of one domain as inflow-BC for another domain
 *
 * The outflow boundary has to be the y-z plane at a given x-coordinate. The velocity is written at
 * n_points_y in y-direction and n_points_z in z-direction, which has to be specified by the user.
 */
enum class InflowGeometry
{
  Cartesian,
  Cylindrical
};

template<int dim>
struct InflowData
{
  InflowData()
    : write_inflow_data(false),
      inflow_geometry(InflowGeometry::Cartesian),
      normal_direction(0),
      normal_coordinate(0.0),
      n_points_y(2),
      n_points_z(2),
      y_values(nullptr),
      z_values(nullptr),
      array(nullptr)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(write_inflow_data == true)
    {
      print_parameter(pcout, "Normal direction", normal_direction);
      print_parameter(pcout, "Normal coordinate", normal_coordinate);
      print_parameter(pcout, "Number of points in y-direction", n_points_y);
      print_parameter(pcout, "Number of points in z-direction", n_points_z);
    }
  }

  // write the data?
  bool write_inflow_data;

  InflowGeometry inflow_geometry;

  // direction normal to inflow surface: has to be 0 (x), 1 (y), or 2 (z)
  unsigned int normal_direction;
  // position of inflow plane in the direction normal to the inflow surface
  double normal_coordinate;
  // specify the number of data points (grid points) in y- and z-direction
  unsigned int n_points_y;
  unsigned int n_points_z;

  // Vectors with the y-coordinates, z-coordinates (in physical space)
  std::vector<double> * y_values;
  std::vector<double> * z_values;
  // and the velocity values at n_points_y*n_points_z points
  std::vector<Tensor<1, dim, double>> * array;
};

// calculation of perturbation energy for Orr-Sommerfeld problem

struct PerturbationEnergyData
{
  PerturbationEnergyData()
    : calculate(false),
      calculate_every_time_steps(std::numeric_limits<unsigned int>::max()),
      filename_prefix("orr_sommerfeld"),
      omega_i(0.0),
      h(1.0),
      U_max(1.0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Calculate perturbation energy:" << std::endl;
      print_parameter(pcout, "Calculate perturbation energy", calculate);
      print_parameter(pcout, "Calculate every timesteps", calculate_every_time_steps);
      print_parameter(pcout, "Filename output", filename_prefix);
      print_parameter(pcout, "Amplification omega_i", omega_i);
      print_parameter(pcout, "Channel height h", h);
      print_parameter(pcout, "Maximum velocity U_max", U_max);
    }
  }

  bool         calculate;
  unsigned int calculate_every_time_steps;
  std::string  filename_prefix;
  double       omega_i;
  double       h;
  double       U_max;
};

template<int dim>
class InputParameters
{
public:
  // standard constructor that initializes parameters
  InputParameters()
    : // MATHEMATICAL MODEL
      problem_type(ProblemType::Undefined),
      equation_type(EquationType::Undefined),
      formulation_viscous_term(FormulationViscousTerm::LaplaceFormulation),
      formulation_convective_term(FormulationConvectiveTerm::DivergenceFormulation),
      use_outflow_bc_convective_term(false),
      right_hand_side(false),

      // PHYSICAL QUANTITIES
      start_time(0.),
      end_time(-1.),
      viscosity(-1.),

      // TEMPORAL DISCRETIZATION
      solver_type(SolverType::Undefined),
      temporal_discretization(TemporalDiscretization::Undefined),
      treatment_of_convective_term(TreatmentOfConvectiveTerm::Undefined),
      time_integrator_oif(TimeIntegratorOIF::Undefined),
      calculation_of_time_step_size(TimeStepCalculation::Undefined),
      adaptive_time_stepping_limiting_factor(1.2),
      max_velocity(-1.),
      cfl(-1.),
      cfl_oif(-1.),
      cfl_exponent_fe_degree_velocity(2.0),
      c_eff(-1.),
      time_step_size(-1.),
      max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
      order_time_integrator(1),
      start_with_low_order(true),

      // pseudo-timestepping
      convergence_criterion_steady_problem(ConvergenceCriterionSteadyProblem::Undefined),
      abs_tol_steady(1.e-20),
      rel_tol_steady(1.e-12),

      // SPATIAL DISCRETIZATION
      degree_mapping(1),
      upwind_factor(1.0),
      type_dirichlet_bc_convective(TypeDirichletBCs::Mirror),

      // convective term - currently no parameters

      // viscous term
      IP_formulation_viscous(InteriorPenaltyFormulation::Undefined),
      penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Undefined),
      IP_factor_viscous(1.),

      // gradient term
      gradp_integrated_by_parts(false),
      gradp_use_boundary_data(false),

      // divergence term
      divu_integrated_by_parts(false),
      divu_use_boundary_data(false),

      // special case: pure DBC's
      pure_dirichlet_bc(false),
      adjust_pressure_level(AdjustPressureLevel::ApplyZeroMeanValue),

      // div-div and continuity penalty
      use_divergence_penalty(false),
      divergence_penalty_factor(1.),
      use_continuity_penalty(false),
      continuity_penalty_components(ContinuityPenaltyComponents::Undefined),
      continuity_penalty_factor(1.),
      type_penalty_parameter(TypePenaltyParameter::Undefined),
      add_penalty_terms_to_monolithic_system(false),

      // NUMERICAL PARAMETERS
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_face_loops(false),

      // PROJECTION METHODS

      // pressure Poisson equation
      IP_factor_pressure(1.),
      solver_pressure_poisson(SolverPressurePoisson::PCG),
      max_n_tmp_vectors_pressure_poisson(30),
      preconditioner_pressure_poisson(PreconditionerPressurePoisson::GeometricMultigrid),
      multigrid_data_pressure_poisson(MultigridData()),
      abs_tol_pressure(1.e-20),
      rel_tol_pressure(1.e-12),

      // stability in the limit of small time steps and projection step
      use_approach_of_ferrer(false),
      deltat_ref(1.0),

      // projection step
      solver_projection(SolverProjection::PCG),
      preconditioner_projection(PreconditionerProjection::InverseMassMatrix),
      preconditioner_block_diagonal_projection(PreconditionerBlockDiagonal::InverseMassMatrix),
      solver_data_block_diagonal_projection(SolverData(1000, 1.e-12, 1.e-2)),
      update_preconditioner_projection(true),
      abs_tol_projection(1.e-20),
      rel_tol_projection(1.e-12),

      // HIGH-ORDER DUAL SPLITTING SCHEME

      // formulations
      order_extrapolation_pressure_nbc((order_time_integrator <= 2) ? order_time_integrator : 2),

      // convective step
      newton_solver_data_convective(NewtonSolverData()),
      abs_tol_linear_convective(1.e-20),
      rel_tol_linear_convective(1.e-12),
      max_iter_linear_convective(std::numeric_limits<unsigned int>::max()),
      use_right_preconditioning_convective(true),
      max_n_tmp_vectors_convective(30),

      // stability in the limit of small time steps and projection step
      small_time_steps_stability(false),

      // viscous step
      solver_viscous(SolverViscous::PCG),
      preconditioner_viscous(PreconditionerViscous::InverseMassMatrix),
      multigrid_data_viscous(MultigridData()),
      abs_tol_viscous(1.e-20),
      rel_tol_viscous(1.e-12),
      update_preconditioner_viscous(false),

      // PRESSURE-CORRECTION SCHEME

      // momentum step
      newton_solver_data_momentum(NewtonSolverData()),
      solver_momentum(SolverMomentum::GMRES),
      preconditioner_momentum(MomentumPreconditioner::Undefined),
      multigrid_data_momentum(MultigridData()),
      abs_tol_momentum_linear(1.e-20),
      rel_tol_momentum_linear(1.e-12),
      max_iter_momentum_linear(std::numeric_limits<unsigned int>::max()),
      use_right_preconditioning_momentum(true),
      max_n_tmp_vectors_momentum(30),
      update_preconditioner_momentum(false),

      // formulations
      order_pressure_extrapolation(1),
      rotational_formulation(false),


      // COUPLED NAVIER-STOKES SOLVER

      // scaling of continuity equation
      use_scaling_continuity(false),
      scaling_factor_continuity(1.0),

      // nonlinear solver (Newton solver)
      newton_solver_data_coupled(NewtonSolverData()),

      // linear solver
      solver_linearized_navier_stokes(SolverLinearizedNavierStokes::Undefined),
      abs_tol_linear(1.e-20),
      rel_tol_linear(1.e-12),
      max_iter_linear(std::numeric_limits<unsigned int>::max()),

      // preconditioning linear solver
      preconditioner_linearized_navier_stokes(PreconditionerLinearizedNavierStokes::Undefined),
      use_right_preconditioning(true),
      max_n_tmp_vectors(30),

      // preconditioner velocity/momentum block
      momentum_preconditioner(MomentumPreconditioner::Undefined),
      multigrid_data_momentum_preconditioner(MultigridData()),
      exact_inversion_of_momentum_block(false),
      rel_tol_solver_momentum_preconditioner(1.e-12),
      max_n_tmp_vectors_solver_momentum_preconditioner(30),

      // preconditioner Schur-complement block
      schur_complement_preconditioner(SchurComplementPreconditioner::Undefined),
      discretization_of_laplacian(DiscretizationOfLaplacian::Undefined),
      multigrid_data_schur_complement_preconditioner(MultigridData()),
      exact_inversion_of_laplace_operator(false),
      rel_tol_solver_schur_complement_preconditioner(1.e-12),

      // update preconditioner
      update_preconditioner(false),

      // TURBULENCE
      use_turbulence_model(false),
      turbulence_model_constant(1.0),
      turbulence_model(TurbulenceEddyViscosityModel::Undefined),
      turb_stat_data(TurbulenceStatisticsData()),
      xwall_turb(XWallTurbulenceApproach::Undefined),
      variabletauw(true),
      dtauw(1.),
      max_wdist_xwall(-1.),
      diffusion_number(-1.),


      // OUTPUT AND POSTPROCESSING

      // print input parameters
      print_input_parameters(false),

      // write output for visualization of results
      output_data(OutputDataNavierStokes()),

      // calculation of error
      error_data(ErrorCalculationData()),

      // output of solver information
      output_solver_info_every_timesteps(1),

      // restart
      restart_data(RestartData()),

      // lift and drag
      lift_and_drag_data(LiftAndDragData()),

      // pressure difference
      pressure_difference_data(PressureDifferenceData<dim>()),

      // conservation of mass
      mass_data(MassConservationData()),

      // turbulent channel statistics
      turb_ch_data(TurbulentChannelData()),

      // inflow data
      inflow_data(InflowData<dim>()),

      // kinetic energy
      kinetic_energy_data(KineticEnergyData()),

      // kinetic energy spectrum
      kinetic_energy_spectrum_data(KineticEnergySpectrumData()),

      // perturbation energy data
      perturbation_energy_data(PerturbationEnergyData()),

      // plot data along line
      line_plot_data(LinePlotData<dim>())
  {
  }

  void
  set_input_parameters();

  void
  set_input_parameters(unsigned int const domain_id);

  void
  check_input_parameters()
  {
    // MATHEMATICAL MODEL
    AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("parameter must be defined"));
    AssertThrow(equation_type != EquationType::Undefined, ExcMessage("parameter must be defined"));

    if(use_outflow_bc_convective_term)
    {
      AssertThrow(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation,
                  ExcMessage("Outflow boundary condition for convective term is currently "
                             "only implemented for divergence formulation of convective term"));
    }

    AssertThrow(formulation_viscous_term != FormulationViscousTerm::Undefined,
                ExcMessage("parameter must be defined"));
    AssertThrow(formulation_convective_term != FormulationConvectiveTerm::Undefined,
                ExcMessage("parameter must be defined"));

    // PHYSICAL QUANTITIES
    AssertThrow(end_time > start_time, ExcMessage("parameter end_time must be defined"));
    AssertThrow(viscosity >= 0.0, ExcMessage("parameter must be defined"));

    // TEMPORAL DISCRETIZATION
    AssertThrow(solver_type != SolverType::Undefined, ExcMessage("parameter must be defined"));
    AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
                ExcMessage("parameter must be defined"));
    AssertThrow(treatment_of_convective_term != TreatmentOfConvectiveTerm::Undefined,
                ExcMessage("parameter must be defined"));
    AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
                ExcMessage("parameter must be defined"));

    if(problem_type == ProblemType::Unsteady)
    {
      AssertThrow(solver_type == SolverType::Unsteady,
                  ExcMessage("An unsteady solver has to be used to solve unsteady problems."));
    }

    if(solver_type == SolverType::Steady)
    {
      AssertThrow(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit,
                  ExcMessage(
                    "Convective term has to be formulated implicitly when using a steady solver."));
    }

    if(problem_type == ProblemType::Steady && solver_type == SolverType::Unsteady)
    {
      if(temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        AssertThrow(
          treatment_of_convective_term != TreatmentOfConvectiveTerm::ExplicitOIF,
          ExcMessage(
            "Operator-integration-factor splitting approach introduces a splitting error. "
            "Hence, this approach cannot be used to solve the steady Navier-Stokes equations."));
      }
    }

    if(calculation_of_time_step_size != TimeStepCalculation::ConstTimeStepUserSpecified)
    {
      AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
      AssertThrow(max_velocity > 0., ExcMessage("parameter must be defined"));
    }
    if(calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
      AssertThrow(time_step_size > 0., ExcMessage("parameter must be defined"));
    if(calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency)
      AssertThrow(c_eff > 0., ExcMessage("parameter must be defined"));


    // SPATIAL DISCRETIZATION
    AssertThrow(degree_mapping > 0, ExcMessage("Invalid parameter."));

    AssertThrow(IP_formulation_viscous != InteriorPenaltyFormulation::Undefined,
                ExcMessage("parameter must be defined"));

    if(formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
      AssertThrow(penalty_term_div_formulation != PenaltyTermDivergenceFormulation::Undefined,
                  ExcMessage("parameter must be defined"));

    if(equation_type == EquationType::NavierStokes)
    {
      AssertThrow(upwind_factor >= 0.0, ExcMessage("Upwind factor must not be negative."));
    }

    if(pure_dirichlet_bc == true)
    {
      if(adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint ||
         adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
      {
        AssertThrow(
          error_data.analytical_solution_available == true,
          ExcMessage(
            "To adjust the pressure level as specified, an analytical solution has to be available."));
      }
    }

    if(use_continuity_penalty == true)
    {
      AssertThrow(continuity_penalty_components != ContinuityPenaltyComponents::Undefined,
                  ExcMessage("Parameter must be defined"));
    }

    if(use_divergence_penalty == true || use_continuity_penalty == true)
    {
      AssertThrow(type_penalty_parameter != TypePenaltyParameter::Undefined,
                  ExcMessage("Parameter must be defined"));
    }

    // HIGH-ORDER DUAL SPLITTING SCHEME
    if(temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      AssertThrow(order_extrapolation_pressure_nbc >= 0 &&
                    order_extrapolation_pressure_nbc <= order_time_integrator,
                  ExcMessage("Invalid input parameter order_extrapolation_pressure_nbc!"));

      if(order_extrapolation_pressure_nbc > 2)
      {
        std::cout
          << "WARNING:" << std::endl
          << "Order of extrapolation of viscous term and convective term in pressure NBC larger than 2 leads to a conditionally stable scheme."
          << std::endl;
      }
    }

    // PRESSURE-CORRECTION SCHEME
    if(temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      AssertThrow(order_pressure_extrapolation >= 0 &&
                    order_pressure_extrapolation <= order_time_integrator,
                  ExcMessage("Invalid input parameter order_pressure_extrapolation!"));

      AssertThrow(preconditioner_momentum != MomentumPreconditioner::Undefined,
                  ExcMessage("parameter must be defined"));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
      {
        AssertThrow(
          preconditioner_momentum != MomentumPreconditioner::VelocityConvectionDiffusion,
          ExcMessage(
            "Use VelocityConvectionDiffusion preconditioner only if convective term is treated implicitly."));
      }
    }

    // COUPLED NAVIER-STOKES SOLVER
    if(temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      if(use_scaling_continuity == true)
        AssertThrow(scaling_factor_continuity > 0.0, ExcMessage("Invalid parameter"));

      AssertThrow(preconditioner_linearized_navier_stokes !=
                    PreconditionerLinearizedNavierStokes::Undefined,
                  ExcMessage("parameter must be defined"));

      AssertThrow(momentum_preconditioner != MomentumPreconditioner::Undefined,
                  ExcMessage("parameter must be defined"));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
      {
        AssertThrow(
          momentum_preconditioner != MomentumPreconditioner::VelocityConvectionDiffusion,
          ExcMessage(
            "Use VelocityConvectionDiffusion preconditioner only if convective term is treated implicitly."));
      }

      AssertThrow(schur_complement_preconditioner != SchurComplementPreconditioner::Undefined,
                  ExcMessage("parameter must be defined"));
      if(schur_complement_preconditioner == SchurComplementPreconditioner::LaplaceOperator ||
         schur_complement_preconditioner == SchurComplementPreconditioner::CahouetChabard ||
         schur_complement_preconditioner == SchurComplementPreconditioner::Elman ||
         schur_complement_preconditioner ==
           SchurComplementPreconditioner::PressureConvectionDiffusion)
      {
        AssertThrow(discretization_of_laplacian != DiscretizationOfLaplacian::Undefined,
                    ExcMessage("parameter must be defined"));
      }
    }

    // OPERATOR-INTEGRATION-FACTOR SPLITTING
    if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      AssertThrow(time_integrator_oif != TimeIntegratorOIF::Undefined,
                  ExcMessage("parameter must be defined"));

      AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
      AssertThrow(cfl_oif > 0., ExcMessage("parameter must be defined"));
    }

    // TURBULENCE
    if(use_turbulence_model)
    {
      AssertThrow(turbulence_model != TurbulenceEddyViscosityModel::Undefined,
                  ExcMessage("parameter must be defined"));
      AssertThrow(turbulence_model_constant > 0, ExcMessage("parameter must be greater than zero"));
    }

    // OUTPUT AND POSTPROCESSING
  }


  void
  print(ConditionalOStream & pcout)
  {
    pcout << std::endl << "List of input parameters:" << std::endl;

    // MATHEMATICAL MODEL
    print_parameters_mathematical_model(pcout);

    // PHYSICAL QUANTITIES
    print_parameters_physical_quantities(pcout);

    // TEMPORAL DISCRETIZATION
    if(solver_type == SolverType::Unsteady)
      print_parameters_temporal_discretization(pcout);

    // SPATIAL DISCRETIZATION
    print_parameters_spatial_discretization(pcout);

    // NUMERICAL PARAMTERS
    print_parameters_numerical_parameters(pcout);

    // HIGH-ORDER DUAL SPLITTING SCHEME
    if(temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      print_parameters_dual_splitting(pcout);

    // PRESSURE-CORRECTION  SCHEME
    if(temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
      print_parameters_pressure_correction(pcout);

    // COUPLED NAVIER-STOKES SOLVER
    if(solver_type == SolverType::Steady ||
       (solver_type == SolverType::Unsteady &&
        temporal_discretization == TemporalDiscretization::BDFCoupledSolution))
      print_parameters_coupled_solver(pcout);

    // TURBULENCE
    print_parameters_turbulence(pcout);

    // OUTPUT AND POSTPROCESSING
    print_parameters_output_and_postprocessing(pcout);
  }


  void
  print_parameters_mathematical_model(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Mathematical model:" << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

    // problem type
    std::string str_problem_type[] = {"Undefined", "Steady", "Unsteady"};

    print_parameter(pcout, "Problem type", str_problem_type[(int)problem_type]);

    // equation type
    std::string str_equation_type[] = {"Undefined", "Stokes", "Navier-Stokes"};

    print_parameter(pcout, "Equation type", str_equation_type[(int)equation_type]);

    // formulation of viscous term
    std::string str_form_viscous_term[] = {"Undefined",
                                           "Divergence formulation",
                                           "Laplace formulation"};

    print_parameter(pcout,
                    "Formulation of viscous term",
                    str_form_viscous_term[(int)formulation_viscous_term]);

    if(equation_type == EquationType::NavierStokes)
    {
      // formulation of convective term
      std::string str_form_convective_term[] = {"Undefined",
                                                "Divergence formulation",
                                                "Convective formulation",
                                                "Energy preserving formulation"};

      print_parameter(pcout,
                      "Formulation of convective term",
                      str_form_convective_term[(int)formulation_convective_term]);

      // outflow BC for convective term
      print_parameter(pcout, "Outflow BC for convective term", use_outflow_bc_convective_term);
    }

    // right hand side
    print_parameter(pcout, "Right-hand side", right_hand_side);
  }


  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;

    // start and end time
    if(problem_type == ProblemType::Unsteady)
    {
      print_parameter(pcout, "Start time", start_time);
      print_parameter(pcout, "End time", end_time);
    }

    // viscosity
    print_parameter(pcout, "Viscosity", viscosity);
  }

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Temporal discretization:" << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

    // temporal discretization scheme
    std::string str_temporal_discretization[] = {"Undefined",
                                                 "BDF dual splitting scheme",
                                                 "BDF pressure-correction scheme",
                                                 "BDF coupled solution"};
    print_parameter(pcout,
                    "Temporal discretization method",
                    str_temporal_discretization[(int)temporal_discretization]);

    // treatment of convective term
    std::string str_conv_term[] = {"Undefined", "Explicit", "ExplicitOIF", "Implicit"};

    print_parameter(pcout,
                    "Treatment of convective term",
                    str_conv_term[(int)treatment_of_convective_term]);

    if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      std::string str_time_int_oif[] = {"Undefined",
                                        "ExplRK1Stage1",
                                        "ExplRK2Stage2",
                                        "ExplRK3Stage3",
                                        "ExplRK4Stage4",
                                        "ExplRK3Stage4Reg2C",
                                        "ExplRK3Stage7Reg2",
                                        "ExplRK4Stage5Reg2C",
                                        "ExplRK4Stage8Reg2",
                                        "ExplRK4Stage5Reg3C",
                                        "ExplRK5Stage9Reg2S"};

      print_parameter(pcout,
                      "Time integrator for OIF splitting",
                      str_time_int_oif[(int)time_integrator_oif]);
    }

    // calculation of time step size
    std::string str_calc_time_step[] = {"Undefined",
                                        "Constant time step (user specified)",
                                        "Constant time step (CFL condition)",
                                        "Adaptive time step (CFL condition)",
                                        "Constant time step (max. efficiency)"};

    print_parameter(pcout,
                    "Calculation of time step size",
                    str_calc_time_step[(int)calculation_of_time_step_size]);

    if(calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
    {
      print_parameter(pcout,
                      "Adaptive time stepping limiting factor",
                      adaptive_time_stepping_limiting_factor);
    }


    // here we do not print quantities such as max_velocity, cfl, time_step_size
    // because this is done by the time integration scheme (or the functions that
    // calculate the time step size)

    // maximum number of time steps
    print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);

    // order of time integration scheme
    print_parameter(pcout, "Order of time integration scheme", order_time_integrator);

    // start time integrator with high or low order method
    print_parameter(pcout, "Start with low order method", start_with_low_order);

    if(problem_type == ProblemType::Steady)
    {
      // treatment of convective term
      std::string str_convergence_crit[] = {"Undefined",
                                            "ResidualSteadyNavierStokes",
                                            "SolutionIncrement"};

      print_parameter(pcout,
                      "Convergence criterion steady problems",
                      str_convergence_crit[(int)convergence_criterion_steady_problem]);

      print_parameter(pcout, "Absolute tolerance", abs_tol_steady);
      print_parameter(pcout, "Relative tolerance", rel_tol_steady);
    }
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial discretization:" << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

    print_parameter(pcout, "Polynomial degree of mapping", degree_mapping);

    if(equation_type == EquationType::NavierStokes)
    {
      print_parameter(pcout, "Convective term - Upwind factor", upwind_factor);

      std::string str_type_dirichlet_convective[] = {"Direct", "Mirror"};

      print_parameter(pcout,
                      "Convective term - Type of Dirichlet BC's",
                      str_type_dirichlet_convective[(int)type_dirichlet_bc_convective]);
    }


    // interior penalty formulation of viscous term
    std::string str_IP_form_visc[] = {"Undefined", "SIPG", "NIPG"};

    print_parameter(pcout,
                    "Viscous term - IP formulation",
                    str_IP_form_visc[(int)IP_formulation_viscous]);

    print_parameter(pcout, "Viscous term - IP factor", IP_factor_viscous);

    if(formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      std::string str_penalty_term_div_form[] = {"Undefined", "Symmetrized", "NotSymmetrized"};

      print_parameter(pcout,
                      "Penalty term formulation viscous term",
                      str_penalty_term_div_form[(int)penalty_term_div_formulation]);
    }

    // pressure gradient term
    print_parameter(pcout, "Grad(p) - integration by parts", gradp_integrated_by_parts);

    print_parameter(pcout, "Grad(p) - use boundary data", gradp_use_boundary_data);

    // divergence term
    print_parameter(pcout, "Div(u) . integration by parts", divu_integrated_by_parts);

    print_parameter(pcout, "Div(u) - use boundary data", divu_use_boundary_data);

    // special case pure DBC's
    print_parameter(pcout, "Pure Dirichlet BC's", pure_dirichlet_bc);

    if(pure_dirichlet_bc == true)
    {
      std::string str_pressure_level[] = {"applying zero mean value",
                                          "applying correct (analytical) mean value",
                                          "applying analytical solution in point"};

      print_parameter(pcout,
                      "Adjust pressure level by",
                      str_pressure_level[(int)adjust_pressure_level]);
    }

    print_parameter(pcout, "Use divergence penalty term", use_divergence_penalty);

    if(use_divergence_penalty == true)
      print_parameter(pcout, "Penalty factor divergence", divergence_penalty_factor);

    print_parameter(pcout, "Use continuity penalty term", use_continuity_penalty);

    if(use_continuity_penalty == true)
    {
      print_parameter(pcout, "Penalty factor continuity", continuity_penalty_factor);

      std::string continuity_penalty_components_str[] = {"Undefined", "All", "Normal"};

      print_parameter(pcout,
                      "Continuity penalty term components",
                      continuity_penalty_components_str[(int)continuity_penalty_components]);
    }

    if(use_divergence_penalty == true || use_continuity_penalty == true)
    {
      std::string type_penalty_parameter_str[] = {"Undefined",
                                                  "ConvectiveTerm",
                                                  "ViscousTerm",
                                                  "ViscousAndConvectiveTerms"};

      print_parameter(pcout,
                      "Type of penalty parameter",
                      type_penalty_parameter_str[(int)type_penalty_parameter]);
    }

    if(temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      if(use_divergence_penalty == true || use_continuity_penalty == true)
        print_parameter(pcout,
                        "Add penalty terms to monolithic system",
                        add_penalty_terms_to_monolithic_system);
    }
  }

  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout)
  {
    print_parameter(pcout,
                    "Block Jacobi matrix-free",
                    implement_block_diagonal_preconditioner_matrix_free);

    print_parameter(pcout, "Use cell-based face loops", use_cell_based_face_loops);
  }

  void
  print_parameters_pressure_poisson(ConditionalOStream & pcout)
  {
    // pressure Poisson equation
    pcout << std::endl << "  Pressure Poisson equation (PPE):" << std::endl;

    print_parameter(pcout, "IP factor PPE", IP_factor_pressure);

    std::string str_solver_ppe[] = {"PCG", "FGMRES"};

    print_parameter(pcout, "Solver PPE", str_solver_ppe[(int)solver_pressure_poisson]);

    if(solver_pressure_poisson == SolverPressurePoisson::FGMRES)
    {
      print_parameter(pcout,
                      "Max number of vectors before restart",
                      max_n_tmp_vectors_pressure_poisson);
    }

    std::string str_precon_ppe[] = {"None", "Jacobi", "GeometricMultigrid"};

    print_parameter(pcout,
                    "Preconditioner PPE",
                    str_precon_ppe[(int)preconditioner_pressure_poisson]);

    if(preconditioner_pressure_poisson == PreconditionerPressurePoisson::GeometricMultigrid)
    {
      multigrid_data_pressure_poisson.print(pcout);
    }

    print_parameter(pcout, "Absolute solver tolerance", abs_tol_pressure);
    print_parameter(pcout, "Relative solver tolerance", rel_tol_pressure);

    // small time steps stability
    print_parameter(pcout, "Approach of Ferrer et al.", use_approach_of_ferrer);
    if(use_approach_of_ferrer == true)
      print_parameter(pcout, "Reference time step size (Ferrer)", deltat_ref);
  }

  void
  print_parameters_projection_step(ConditionalOStream & pcout)
  {
    if(use_divergence_penalty == true)
    {
      std::string str_solver_proj[] = {"LU", "PCG"};

      print_parameter(pcout, "Solver projection step", str_solver_proj[(int)solver_projection]);

      if(use_divergence_penalty == true && use_continuity_penalty == true)
      {
        std::string str_precon_proj[] = {"None", "PointJacobi", "BlockJacobi", "InverseMassMatrix"};

        print_parameter(pcout,
                        "Preconditioner projection step",
                        str_precon_proj[(int)preconditioner_projection]);

        if(preconditioner_projection == PreconditionerProjection::BlockJacobi &&
           implement_block_diagonal_preconditioner_matrix_free)
        {
          std::string str_precon[] = {"Undefined", "None", "InverseMassMatrix"};

          print_parameter(pcout,
                          "Preconditioner block diagonal",
                          str_precon[(int)preconditioner_block_diagonal_projection]);

          solver_data_block_diagonal_projection.print(pcout);
        }

        print_parameter(pcout,
                        "Update preconditioner projection step",
                        update_preconditioner_projection);
      }

      print_parameter(pcout, "Absolute solver tolerance", abs_tol_projection);
      print_parameter(pcout, "Relative solver tolerance", rel_tol_projection);
    }
  }

  void
  print_parameters_dual_splitting(ConditionalOStream & pcout)
  {
    pcout << std::endl << "High-order dual splitting scheme:" << std::endl;

    // formulations
    print_parameter(pcout, "Order of extrapolation pressure NBC", order_extrapolation_pressure_nbc);

    // convective step
    pcout << "  Convective step:" << std::endl;

    // Newton solver
    if(equation_type == EquationType::NavierStokes &&
       treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    {
      pcout << "  Newton solver:" << std::endl;

      print_parameter(pcout, "Absolute solver tolerance", newton_solver_data_convective.abs_tol);
      print_parameter(pcout, "Relative solver tolerance", newton_solver_data_convective.rel_tol);
      print_parameter(pcout,
                      "Maximum number of iterations",
                      newton_solver_data_convective.max_iter);

      pcout << "  Linear solver:" << std::endl;

      print_parameter(pcout, "Absolute solver tolerance", abs_tol_linear_convective);
      print_parameter(pcout, "Relative solver tolerance", rel_tol_linear_convective);
      print_parameter(pcout, "Maximum number of iterations", max_iter_linear_convective);
      print_parameter(pcout, "Right preconditioning", use_right_preconditioning_convective);
      print_parameter(pcout, "Max number of vectors before restart", max_n_tmp_vectors_convective);
    }



    // small time steps stability
    pcout << std::endl << "  Small time steps stability:" << std::endl;

    print_parameter(pcout, "STS stability approach", small_time_steps_stability);

    // projection method
    print_parameters_pressure_poisson(pcout);

    // projection step
    pcout << std::endl << "  Projection step:" << std::endl;
    print_parameters_projection_step(pcout);

    // Viscous step
    pcout << std::endl << "  Viscous step:" << std::endl;

    std::string str_solver_viscous[] = {"PCG", "GMRES", "FGMRES"};

    print_parameter(pcout, "Solver viscous step", str_solver_viscous[(int)solver_viscous]);

    std::string str_precon_viscous[] = {
      "None", "InverseMassMatrix", "PointJacobi", "BlockJacobi", "GeometricMultigrid"};

    print_parameter(pcout,
                    "Preconditioner viscous step",
                    str_precon_viscous[(int)preconditioner_viscous]);

    if(preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      multigrid_data_viscous.print(pcout);
    }

    print_parameter(pcout, "Absolute solver tolerance", abs_tol_viscous);
    print_parameter(pcout, "Relative solver tolerance", rel_tol_viscous);

    print_parameter(pcout, "Udpate preconditioner viscous", update_preconditioner_viscous);
  }

  void
  print_parameters_pressure_correction(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Pressure-correction scheme:" << std::endl;

    // Momentum step
    pcout << "  Momentum step:" << std::endl;

    // Newton solver
    if(equation_type == EquationType::NavierStokes &&
       treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    {
      pcout << "  Newton solver:" << std::endl;

      print_parameter(pcout, "Absolute solver tolerance", newton_solver_data_momentum.abs_tol);
      print_parameter(pcout, "Relative solver tolerance", newton_solver_data_momentum.rel_tol);
      print_parameter(pcout, "Maximum number of iterations", newton_solver_data_momentum.max_iter);

      pcout << std::endl;
    }

    // Solver linearized problem
    pcout << "  Linear solver:" << std::endl;

    std::string str_solver_momentum[] = {"PCG", "GMRES", "FGMRES"};

    print_parameter(pcout,
                    "Solver for linear(ized) problem",
                    str_solver_momentum[(int)solver_momentum]);

    std::string str_precon_momentum[] = {"Undefined",
                                         "None",
                                         "PointJacobi",
                                         "BlockJacobi",
                                         "InverseMassMatrix",
                                         "VelocityDiffusion",
                                         "VelocityConvectionDiffusion"};

    print_parameter(pcout,
                    "Preconditioner linear(ized) problem",
                    str_precon_momentum[(int)preconditioner_momentum]);

    print_parameter(pcout, "Absolute solver tolerance", abs_tol_momentum_linear);
    print_parameter(pcout, "Relative solver tolerance", rel_tol_momentum_linear);
    print_parameter(pcout, "Maximum number of iterations", max_iter_momentum_linear);
    print_parameter(pcout, "Right preconditioning", use_right_preconditioning_momentum);

    if(solver_momentum == SolverMomentum::GMRES)
      print_parameter(pcout, "Max number of vectors before restart", max_n_tmp_vectors_momentum);

    print_parameter(pcout, "Update of preconditioner", update_preconditioner_momentum);

    // formulations of pressur-correction scheme
    pcout << std::endl << "  Formulation of pressure-correction scheme:" << std::endl;
    print_parameter(pcout, "Order of pressure extrapolation", order_pressure_extrapolation);
    print_parameter(pcout, "Rotational formulation", rotational_formulation);

    // projection method
    print_parameters_pressure_poisson(pcout);

    // projection step
    pcout << std::endl << "  Projection step:" << std::endl;
    print_parameters_projection_step(pcout);
  }


  void
  print_parameters_coupled_solver(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Coupled Navier-Stokes solver:" << std::endl;

    print_parameter(pcout, "Use scaling of continuity equation", use_scaling_continuity);
    if(use_scaling_continuity == true)
      print_parameter(pcout, "Scaling factor continuity equation", scaling_factor_continuity);

    pcout << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

    // Newton solver

    // if a nonlinear problem has to be solved
    if(equation_type == EquationType::NavierStokes &&
       (problem_type == ProblemType::Steady ||
        treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit))
    {
      pcout << "Newton solver:" << std::endl;

      print_parameter(pcout, "Absolute solver tolerance", newton_solver_data_coupled.abs_tol);
      print_parameter(pcout, "Relative solver tolerance", newton_solver_data_coupled.rel_tol);
      print_parameter(pcout, "Maximum number of iterations", newton_solver_data_coupled.max_iter);

      pcout << std::endl;
    }

    // Solver linearized problem
    pcout << "Linear solver:" << std::endl;

    std::string str_solver_linearized[] = {"Undefined", "GMRES", "FGMRES"};

    print_parameter(pcout,
                    "Solver for linear(ized) problem",
                    str_solver_linearized[(int)solver_linearized_navier_stokes]);

    print_parameter(pcout, "Absolute solver tolerance", abs_tol_linear);
    print_parameter(pcout, "Relative solver tolerance", rel_tol_linear);
    print_parameter(pcout, "Maximum number of iterations", max_iter_linear);

    std::string str_precon_linear[] = {
      "Undefined", "None", "BlockDiagonal", "BlockTriangular", "BlockTriangularFactorization"};

    print_parameter(pcout,
                    "Preconditioner linear(ized) problem",
                    str_precon_linear[(int)preconditioner_linearized_navier_stokes]);

    print_parameter(pcout, "Right preconditioning", use_right_preconditioning);

    if(solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES)
      print_parameter(pcout, "Max number of vectors before restart", max_n_tmp_vectors);

    // preconditioner momentum block
    std::string str_momentum_precon[] = {"Undefined",
                                         "None",
                                         "PointJacobi",
                                         "BlockJacobi",
                                         "InverseMassMatrix",
                                         "VelocityDiffusion",
                                         "VelocityConvectionDiffusion"};

    print_parameter(pcout,
                    "Preconditioner momentum block",
                    str_momentum_precon[(int)momentum_preconditioner]);

    if(momentum_preconditioner == MomentumPreconditioner::VelocityDiffusion ||
       momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion)
    {
      multigrid_data_momentum_preconditioner.print(pcout);

      print_parameter(pcout,
                      "Exact inversion of momentum block",
                      exact_inversion_of_momentum_block);

      if(exact_inversion_of_momentum_block == true)
      {
        print_parameter(pcout, "Relative solver tolerance", rel_tol_solver_momentum_preconditioner);

        print_parameter(pcout,
                        "Max number of vectors before restart",
                        max_n_tmp_vectors_solver_momentum_preconditioner);
      }

      print_parameter(pcout, "Update preconditioner", update_preconditioner);
    }

    // preconditioner Schur-complement block
    std::string str_schur_precon[] = {"Undefined",
                                      "None",
                                      "InverseMassMatrix",
                                      "LaplaceOperator",
                                      "CahouetChabard",
                                      "Elman",
                                      "PressureConvectionDiffusion"};

    print_parameter(pcout,
                    "Schur-complement preconditioner",
                    str_schur_precon[(int)schur_complement_preconditioner]);

    if(schur_complement_preconditioner == SchurComplementPreconditioner::LaplaceOperator ||
       schur_complement_preconditioner == SchurComplementPreconditioner::CahouetChabard ||
       schur_complement_preconditioner == SchurComplementPreconditioner::Elman ||
       schur_complement_preconditioner ==
         SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      std::string str_discret_laplacian[] = {"Undefined", "Classical", "Compatible"};
      print_parameter(pcout,
                      "Discretization of Laplacian",
                      str_discret_laplacian[(int)discretization_of_laplacian]);

      multigrid_data_schur_complement_preconditioner.print(pcout);

      print_parameter(pcout,
                      "Exact inversion of Laplace operator",
                      exact_inversion_of_laplace_operator);

      if(exact_inversion_of_laplace_operator)
      {
        print_parameter(pcout,
                        "Relative solver tolerance",
                        rel_tol_solver_schur_complement_preconditioner);
      }
    }

    // projection_step
    if(use_divergence_penalty == true || use_continuity_penalty == true)
    {
      pcout << std::endl << "Postprocessing of velocity:" << std::endl;
      print_parameters_projection_step(pcout);
    }
  }

  void
  print_parameters_turbulence(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Turbulence:" << std::endl;

    print_parameter(pcout, "Use turbulence model", use_turbulence_model);

    if(use_turbulence_model == true)
    {
      std::string str_turbulence_model[] = {"Undefined", "Smagorinsky", "Vreman", "WALE", "Sigma"};

      print_parameter(pcout, "Turbulence model", str_turbulence_model[(int)turbulence_model]);
      print_parameter(pcout, "Turbulence model constant", turbulence_model_constant);
    }

    if(false)
    {
      std::string str_xwall_turbulence_approach[] = {
        "Undefined", "None", "RANSSpalartAllmaras", "ClassicalDES", "MultiscaleDES"};

      print_parameter(pcout,
                      "Turbulence model for xwall",
                      str_xwall_turbulence_approach[(int)xwall_turb]);
    }
  }

  void
  print_parameters_output_and_postprocessing(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Output and postprocessing:" << std::endl;

    // output for visualization of results
    output_data.print(pcout, problem_type == ProblemType::Unsteady);

    // calculation of error
    error_data.print(pcout, problem_type == ProblemType::Unsteady);

    // output of solver information
    if(problem_type == ProblemType::Unsteady)
    {
      print_parameter(pcout,
                      "Output solver info every timesteps",
                      output_solver_info_every_timesteps);
    }

    // restart
    if(problem_type == ProblemType::Unsteady)
    {
      restart_data.print(pcout);
    }

    // turbulent channel statistics
    turb_ch_data.print(pcout);

    // inflow data
    inflow_data.print(pcout);

    // kinetic energy
    kinetic_energy_data.print(pcout);

    // kinetic energy spectrum
    kinetic_energy_spectrum_data.print(pcout);

    // perturbation energy
    perturbation_energy_data.print(pcout);
  }

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
  bool right_hand_side;



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

  // This parameter defines by which factor the time step size is allowed to increase
  // or to decrease in case of adaptive time step, e.g., if one wants to avoid large
  // jumps in the time step size. A factor of 1 implies that the time step size can not
  // change at all, while a factor towards infinity implies that arbitrary changes in
  // the time step size are allowed from one time step to the next.
  double adaptive_time_stepping_limiting_factor;

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


  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Polynomial degree of shape functions used for geometry approximation (mapping from
  // parameter space to physical space)
  unsigned int degree_mapping;

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

  // use boundary data if integrated by parts
  bool gradp_use_boundary_data;

  // integration by parts of div(U)
  bool divu_integrated_by_parts;

  // use boundary data if integrated by parts
  bool divu_use_boundary_data;

  // special case of pure Dirichlet BCs on whole boundary
  bool pure_dirichlet_bc;

  // adjust pressure level in case of pure Dirichlet BC's where
  // the pressure is only defined up to an additive constant
  AdjustPressureLevel adjust_pressure_level;

  // use div-div penalty term
  // Note that this parameter is currently only relevant for the coupled solver
  bool use_divergence_penalty;

  // penalty factor of divergence penalty term
  double divergence_penalty_factor;

  // use continuity penalty term
  // Note that this parameter is currently only relevant for the coupled solver
  bool use_continuity_penalty;

  // specify which components of the velocity field to penalize, i.e., normal
  // components only or all components
  ContinuityPenaltyComponents continuity_penalty_components;

  // penalty factor of continuity penalty term
  double continuity_penalty_factor;

  // type of penalty parameter (see enum declarationn for more information
  TypePenaltyParameter type_penalty_parameter;

  // Add divergence and continuity penalty terms to monolithic system of equations.
  // This parameter is only relevant for the coupled solution approach but not for
  // the projection-type solution methods.
  bool add_penalty_terms_to_monolithic_system;

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

  // defines the maximum size of the Krylov subspace before restart
  unsigned int max_n_tmp_vectors_pressure_poisson;

  // description: see enum declaration
  PreconditionerPressurePoisson preconditioner_pressure_poisson;

  // description: see declaration of MultigridData
  MultigridData multigrid_data_pressure_poisson;

  // solver tolerances for pressure Poisson equation
  double abs_tol_pressure;
  double rel_tol_pressure;

  // use approach of Ferrer et al. (increase penalty parameter when reducing
  // the time step in order to improve stability in the limit of small time steps)
  bool use_approach_of_ferrer;

  // reference time step size that is used when use_approach_of_ferrer == true
  double deltat_ref;

  // PROJECTION STEP

  // description: see enum declaration
  SolverProjection solver_projection;

  // description: see enum declaration
  PreconditionerProjection preconditioner_projection;

  // description: see enum declaration (only relevant if block diagonal is used as
  // preconditioner)
  PreconditionerBlockDiagonal preconditioner_block_diagonal_projection;

  // solver data for block Jacobi preconditioner (only relevant if elementwise
  // iterative solution procedure is used for block diagonal preconditioner)
  SolverData solver_data_block_diagonal_projection;

  // Update preconditioner before solving the linear system of equations.
  // Note that this variable is only used when using an iterative method
  // to solve the global projection equation.
  bool update_preconditioner_projection;

  // solver tolerances for projection step
  double abs_tol_projection;
  double rel_tol_projection;


  /**************************************************************************************/
  /*                                                                                    */
  /*                        HIGH-ORDER DUAL SPLITTING SCHEME                            */
  /*                                                                                    */
  /**************************************************************************************/

  // FORMULATIONS

  // order of extrapolation of viscous term and convective term in pressure Neumann BC
  unsigned int order_extrapolation_pressure_nbc;

  // CONVECTIVE STEP
  NewtonSolverData newton_solver_data_convective;

  // linear solver tolerances for momentum equation
  double       abs_tol_linear_convective;
  double       rel_tol_linear_convective;
  unsigned int max_iter_linear_convective;

  // use right preconditioning
  bool use_right_preconditioning_convective;

  // defines the maximum size of the Krylov subspace before restart
  unsigned int max_n_tmp_vectors_convective;


  // SMALL TIME STEPS: use small time steps stability approach
  // (similar to approach of Leriche et al.)
  bool small_time_steps_stability;

  // VISCOUS STEP

  // description: see enum declaration
  SolverViscous solver_viscous;

  // description: see enum declaration
  PreconditionerViscous preconditioner_viscous;

  // description: see declaration of MultigridData
  MultigridData multigrid_data_viscous;

  // solver tolerances for Helmholtz equation of viscous step
  double abs_tol_viscous;
  double rel_tol_viscous;

  // update preconditioner before every solve of the viscous step
  bool update_preconditioner_viscous;


  /**************************************************************************************/
  /*                                                                                    */
  /*                            PRESSURE-CORRECTION SCHEME                              */
  /*                                                                                    */
  /**************************************************************************************/

  // Newton solver data
  NewtonSolverData newton_solver_data_momentum;

  // description: see enum declaration
  SolverMomentum solver_momentum;

  // description: see enum declaration
  MomentumPreconditioner preconditioner_momentum;

  // description: see declaration of MultigridData
  MultigridData multigrid_data_momentum;

  // linear solver tolerances for momentum equation
  double       abs_tol_momentum_linear;
  double       rel_tol_momentum_linear;
  unsigned int max_iter_momentum_linear;

  // use right preconditioning
  bool use_right_preconditioning_momentum;

  // defines the maximum size of the Krylov subspace before restart
  unsigned int max_n_tmp_vectors_momentum;

  // update preconditioner before solving the linear system of equations
  // only necessary if the parts of the operator change during the simulation
  bool update_preconditioner_momentum;

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
  SolverLinearizedNavierStokes solver_linearized_navier_stokes;

  // solver tolerances for linearized problem of Newton solver
  double       abs_tol_linear;
  double       rel_tol_linear;
  unsigned int max_iter_linear;

  // description: see enum declaration
  PreconditionerLinearizedNavierStokes preconditioner_linearized_navier_stokes;

  // use right preconditioning
  bool use_right_preconditioning;

  // defines the maximum size of the Krylov subspace before restart
  unsigned int max_n_tmp_vectors;

  // description: see enum declaration
  MomentumPreconditioner momentum_preconditioner;

  // description: see declaration
  MultigridData multigrid_data_momentum_preconditioner;

  // The momentum block is inverted "exactly" in block preconditioner
  // by solving the velocity convection-diffusion problem to a given
  // relative tolerance
  bool exact_inversion_of_momentum_block;

  // relative tolerance for solver_momentum_preconditioner
  double rel_tol_solver_momentum_preconditioner;

  // defines the maximum size of the Krylov subspace before restart
  // (for solver of momentum equation in block preconditioner)
  unsigned int max_n_tmp_vectors_solver_momentum_preconditioner;

  // description: see enum declaration
  SchurComplementPreconditioner schur_complement_preconditioner;

  // description: see enum declaration
  DiscretizationOfLaplacian discretization_of_laplacian;

  // description: see declaration
  MultigridData multigrid_data_schur_complement_preconditioner;

  // The Laplace operator is inverted "exactly" in block preconditioner
  // by solving the Laplace problem to a given relative tolerance
  bool exact_inversion_of_laplace_operator;

  // relative tolerance for solver_schur_complement_preconditioner
  double rel_tol_solver_schur_complement_preconditioner;

  // Update preconditioner
  bool update_preconditioner;

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

  // turublence parameters that are required for statistics (post-processing)
  TurbulenceStatisticsData turb_stat_data;

  // turbulence approach for xwall
  XWallTurbulenceApproach xwall_turb;

  // xwall with adaptive wall shear stress
  bool variabletauw;

  // delta tauw if adaptive between 0 and 1
  double dtauw;

  // max wall distance of enriched elements
  double max_wdist_xwall;

  // diffusion number: used to define time step size for the Spalart Allmaras equations
  double diffusion_number;


  /**************************************************************************************/
  /*                                                                                    */
  /*                               OUTPUT AND POSTPROCESSING                            */
  /*                                                                                    */
  /**************************************************************************************/

  // print input parameters at the beginning of the simulation
  bool print_input_parameters;

  // writing output for visualization
  OutputDataNavierStokes output_data;

  // calculating errors
  ErrorCalculationData error_data;

  // show solver performance (wall time, number of iterations) every ... timesteps
  unsigned int output_solver_info_every_timesteps;

  // restart
  RestartData restart_data;

  // computation of lift and drag coefficients
  LiftAndDragData lift_and_drag_data;

  // computation of pressure difference between two points
  PressureDifferenceData<dim> pressure_difference_data;

  // analysis of mass conservation
  MassConservationData mass_data;

  // turbulent channel statistics
  TurbulentChannelData turb_ch_data;

  // inflow data
  InflowData<dim> inflow_data;

  // kinetic energy
  KineticEnergyData kinetic_energy_data;

  // kinetic energy spectrum
  KineticEnergySpectrumData kinetic_energy_spectrum_data;

  // perturbation energy data (Orr-Sommerfeld)
  PerturbationEnergyData perturbation_energy_data;

  // plot along lines
  LinePlotData<dim> line_plot_data;

  // mean flow
  MeanVelocityCalculatorData<dim> mean_velocity_data;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_ */
