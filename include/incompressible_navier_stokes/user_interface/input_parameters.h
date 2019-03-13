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
#include "../../functionalities/solver_info_data.h"
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

#include "enum_types.h"

namespace IncNS
{
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
      adaptive_time_stepping(false),
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

      // pseudo time-stepping
      convergence_criterion_steady_problem(ConvergenceCriterionSteadyProblem::Undefined),
      abs_tol_steady(1.e-20),
      rel_tol_steady(1.e-12),

      // SPATIAL DISCRETIZATION

      // triangulation
      triangulation_type(TriangulationType::Undefined),

      // mapping
      degree_mapping(1),

      // convective term
      upwind_factor(1.0),
      type_dirichlet_bc_convective(TypeDirichletBCs::Mirror),

      // viscous term
      IP_formulation_viscous(InteriorPenaltyFormulation::Undefined),
      penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Symmetrized),
      IP_factor_viscous(1.),

      // gradient term
      gradp_integrated_by_parts(true),
      gradp_use_boundary_data(true),

      // divergence term
      divu_integrated_by_parts(true),
      divu_use_boundary_data(true),

      // special case: pure DBC's
      pure_dirichlet_bc(false),
      adjust_pressure_level(AdjustPressureLevel::ApplyZeroMeanValue),

      // div-div and continuity penalty terms
      use_divergence_penalty(true),
      divergence_penalty_factor(1.),
      use_continuity_penalty(true),
      continuity_penalty_components(ContinuityPenaltyComponents::Normal),
      continuity_penalty_factor(1.),
      type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      add_penalty_terms_to_monolithic_system(false),

      // NUMERICAL PARAMETERS
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_face_loops(false),

      // PROJECTION METHODS

      // pressure Poisson equation
      IP_factor_pressure(1.),
      solver_pressure_poisson(SolverPressurePoisson::CG),
      solver_data_pressure_poisson(SolverData(1e4, 1.e-12, 1.e-6, 100)),
      preconditioner_pressure_poisson(PreconditionerPressurePoisson::Multigrid),
      multigrid_data_pressure_poisson(MultigridData()),

      // projection step
      solver_projection(SolverProjection::CG),
      solver_data_projection(SolverData(1000, 1.e-12, 1.e-6, 100)),
      preconditioner_projection(PreconditionerProjection::InverseMassMatrix),
      update_preconditioner_projection(true),
      update_preconditioner_projection_every_time_steps(1),
      preconditioner_block_diagonal_projection(PreconditionerBlockDiagonal::InverseMassMatrix),
      solver_data_block_diagonal_projection(SolverData(1000, 1.e-12, 1.e-2, 1000)),

      // HIGH-ORDER DUAL SPLITTING SCHEME

      // formulations
      order_extrapolation_pressure_nbc((order_time_integrator <= 2) ? order_time_integrator : 2),

      // convective step
      newton_solver_data_convective(NewtonSolverData(1e2, 1.e-12, 1.e-6)),
      solver_data_convective(SolverData(1e4, 1.e-12, 1.e-6, 100)),

      // viscous step
      solver_viscous(SolverViscous::CG),
      solver_data_viscous(SolverData(1e4, 1.e-12, 1.e-6, 100)),
      preconditioner_viscous(PreconditionerViscous::InverseMassMatrix),
      update_preconditioner_viscous(false),
      update_preconditioner_viscous_every_time_steps(1),
      multigrid_data_viscous(MultigridData()),

      // PRESSURE-CORRECTION SCHEME

      // momentum step
      newton_solver_data_momentum(NewtonSolverData(1e2, 1.e-12, 1.e-6)),
      solver_momentum(SolverMomentum::GMRES),
      solver_data_momentum(SolverData(1e4, 1.e-12, 1.e-6, 100)),
      preconditioner_momentum(MomentumPreconditioner::InverseMassMatrix),
      update_preconditioner_momentum(false),
      update_preconditioner_momentum_every_newton_iter(1),
      update_preconditioner_momentum_every_time_steps(1),
      multigrid_data_momentum(MultigridData()),
      multigrid_operator_type_momentum(MultigridOperatorType::Undefined),

      // formulations
      order_pressure_extrapolation(1),
      rotational_formulation(false),


      // COUPLED NAVIER-STOKES SOLVER

      // scaling of continuity equation
      use_scaling_continuity(false),
      scaling_factor_continuity(1.0),

      // nonlinear solver (Newton solver)
      newton_solver_data_coupled(NewtonSolverData(1e2, 1.e-12, 1.e-6)),

      // linear solver
      solver_coupled(SolverCoupled::GMRES),
      solver_data_coupled(SolverData(1e4, 1.e-12, 1.e-6, 100)),

      // preconditioning linear solver
      preconditioner_coupled(PreconditionerCoupled::BlockTriangular),
      update_preconditioner_coupled(false),
      update_preconditioner_coupled_every_newton_iter(1),
      update_preconditioner_coupled_every_time_steps(1),

      // preconditioner velocity/momentum block
      preconditioner_velocity_block(MomentumPreconditioner::InverseMassMatrix),
      multigrid_operator_type_velocity_block(MultigridOperatorType::Undefined),
      multigrid_data_velocity_block(MultigridData()),
      exact_inversion_of_velocity_block(false),
      solver_data_velocity_block(SolverData(1e4, 1.e-12, 1.e-6, 100)),

      // preconditioner pressure/Schur-complement block
      preconditioner_pressure_block(SchurComplementPreconditioner::PressureConvectionDiffusion),
      discretization_of_laplacian(DiscretizationOfLaplacian::Classical),
      multigrid_data_pressure_block(MultigridData()),
      exact_inversion_of_laplace_operator(false),
      solver_data_pressure_block(SolverData(1e4, 1.e-12, 1.e-6, 100)),

      // TURBULENCE
      use_turbulence_model(false),
      turbulence_model_constant(1.0),
      turbulence_model(TurbulenceEddyViscosityModel::Undefined),
      turb_stat_data(TurbulenceStatisticsData()),

      // OUTPUT AND POSTPROCESSING

      // print input parameters
      print_input_parameters(true),

      // write output for visualization of results
      output_data(OutputDataNavierStokes()),

      // calculation of error
      error_data(ErrorCalculationData()),

      // output of solver information
      solver_info_data(SolverInfoData()),

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
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    // MATHEMATICAL MODEL
    AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("parameter must be defined"));
    AssertThrow(equation_type != EquationType::Undefined, ExcMessage("parameter must be defined"));

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

    if(calculation_of_time_step_size == TimeStepCalculation::CFL)
    {
      AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
      AssertThrow(max_velocity > 0., ExcMessage("parameter must be defined"));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
      AssertThrow(time_step_size > 0., ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
      AssertThrow(c_eff > 0., ExcMessage("parameter must be defined"));

    if(adaptive_time_stepping)
    {
      AssertThrow(calculation_of_time_step_size == TimeStepCalculation::CFL,
                  ExcMessage(
                    "Adaptive time stepping is only implemented for TimeStepCalculation::CFL."));
    }

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

    // SPATIAL DISCRETIZATION
    AssertThrow(triangulation_type != TriangulationType::Undefined,
                ExcMessage("parameter must be defined"));

    AssertThrow(degree_mapping > 0, ExcMessage("Invalid parameter."));

    AssertThrow(IP_formulation_viscous != InteriorPenaltyFormulation::Undefined,
                ExcMessage("parameter must be defined"));

    if(formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      AssertThrow(penalty_term_div_formulation != PenaltyTermDivergenceFormulation::Undefined,
                  ExcMessage("parameter must be defined"));
    }

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
        pcout
          << "WARNING:" << std::endl
          << "Order of extrapolation of viscous and convective terms in pressure Neumann boundary "
          << "condition is larger than 2 which leads to a conditionally stable scheme."
          << std::endl;
      }

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      {
        pcout << "WARNING:" << std::endl
              << "An implicit treatment of the convective term in combination with the "
              << "dual splitting projection scheme is only first order accurate in time."
              << std::endl;
      }
    }

    // PRESSURE-CORRECTION SCHEME
    if(temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      AssertThrow(order_pressure_extrapolation >= 0 &&
                    order_pressure_extrapolation <= order_time_integrator,
                  ExcMessage("Invalid input parameter order_pressure_extrapolation!"));

      if(preconditioner_momentum == MomentumPreconditioner::Multigrid)
      {
        AssertThrow(multigrid_operator_type_momentum != MultigridOperatorType::Undefined,
                    ExcMessage("Parameter must be defined"));

        if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
        {
          AssertThrow(multigrid_operator_type_momentum !=
                        MultigridOperatorType::ReactionConvectionDiffusion,
                      ExcMessage("Invalid parameter. Convective term is treated explicitly."));
        }
      }
    }

    // COUPLED NAVIER-STOKES SOLVER
    if(temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      if(use_scaling_continuity == true)
        AssertThrow(scaling_factor_continuity > 0.0, ExcMessage("Invalid parameter"));

      if(preconditioner_velocity_block == MomentumPreconditioner::Multigrid)
      {
        AssertThrow(multigrid_operator_type_velocity_block != MultigridOperatorType::Undefined,
                    ExcMessage("Parameter must be defined"));

        if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
        {
          AssertThrow(multigrid_operator_type_velocity_block !=
                        MultigridOperatorType::ReactionConvectionDiffusion,
                      ExcMessage("Invalid parameter. Convective term is treated explicitly."));
        }
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

    // NUMERICAL PARAMETERS
    if(implement_block_diagonal_preconditioner_matrix_free)
    {
      AssertThrow(
        use_cell_based_face_loops == true,
        ExcMessage(
          "Cell based face loops have to be used for matrix-free implementation of block diagonal preconditioner."));
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

    print_parameter(pcout, "Problem type", enum_to_string(problem_type));
    print_parameter(pcout, "Equation type", enum_to_string(equation_type));
    print_parameter(pcout, "Formulation of viscous term", enum_to_string(formulation_viscous_term));

    if(equation_type == EquationType::NavierStokes)
    {
      print_parameter(pcout,
                      "Formulation of convective term",
                      enum_to_string(formulation_convective_term));
      print_parameter(pcout, "Outflow BC for convective term", use_outflow_bc_convective_term);
    }

    print_parameter(pcout, "Right-hand side", right_hand_side);
  }


  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;

    // start and end time
    if(solver_type == SolverType::Unsteady)
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

    print_parameter(pcout,
                    "Temporal discretization method",
                    enum_to_string(temporal_discretization));
    print_parameter(pcout,
                    "Treatment of convective term",
                    enum_to_string(treatment_of_convective_term));

    if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      print_parameter(pcout,
                      "Time integrator for OIF splitting",
                      enum_to_string(time_integrator_oif));
    }

    print_parameter(pcout,
                    "Calculation of time step size",
                    enum_to_string(calculation_of_time_step_size));

    print_parameter(pcout, "Adaptive time stepping", adaptive_time_stepping);

    if(adaptive_time_stepping)
    {
      print_parameter(pcout,
                      "Adaptive time stepping limiting factor",
                      adaptive_time_stepping_limiting_factor);
    }


    // here we do not print quantities such as max_velocity, cfl, time_step_size
    // because this is done by the time integration scheme (or the functions that
    // calculate the time step size)

    print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);
    print_parameter(pcout, "Order of time integration scheme", order_time_integrator);
    print_parameter(pcout, "Start with low order method", start_with_low_order);

    if(problem_type == ProblemType::Steady)
    {
      print_parameter(pcout,
                      "Convergence criterion steady problems",
                      enum_to_string(convergence_criterion_steady_problem));

      print_parameter(pcout, "Absolute tolerance", abs_tol_steady);
      print_parameter(pcout, "Relative tolerance", rel_tol_steady);
    }
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial discretization:" << std::endl;

    print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));

    print_parameter(pcout, "Polynomial degree of mapping", degree_mapping);

    if(equation_type == EquationType::NavierStokes)
    {
      print_parameter(pcout, "Convective term - Upwind factor", upwind_factor);
      print_parameter(pcout,
                      "Convective term - Type of Dirichlet BC's",
                      enum_to_string(type_dirichlet_bc_convective));
    }

    print_parameter(pcout, "Viscous term - IP formulation", enum_to_string(IP_formulation_viscous));
    print_parameter(pcout, "Viscous term - IP factor", IP_factor_viscous);

    if(formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      print_parameter(pcout,
                      "Penalty term formulation viscous term",
                      enum_to_string(penalty_term_div_formulation));
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
      print_parameter(pcout, "Adjust pressure level", enum_to_string(adjust_pressure_level));
    }

    print_parameter(pcout, "Use divergence penalty term", use_divergence_penalty);

    if(use_divergence_penalty == true)
      print_parameter(pcout, "Penalty factor divergence", divergence_penalty_factor);

    print_parameter(pcout, "Use continuity penalty term", use_continuity_penalty);

    if(use_continuity_penalty == true)
    {
      print_parameter(pcout, "Penalty factor continuity", continuity_penalty_factor);

      print_parameter(pcout,
                      "Continuity penalty term components",
                      enum_to_string(continuity_penalty_components));
    }

    if(use_divergence_penalty == true || use_continuity_penalty == true)
    {
      print_parameter(pcout, "Type of penalty parameter", enum_to_string(type_penalty_parameter));
    }

    if(temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      if(use_divergence_penalty == true || use_continuity_penalty == true)
      {
        print_parameter(pcout,
                        "Add penalty terms to monolithic system",
                        add_penalty_terms_to_monolithic_system);
      }
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

    print_parameter(pcout, "interior penalty factor", IP_factor_pressure);

    print_parameter(pcout, "Solver", enum_to_string(solver_pressure_poisson));

    solver_data_pressure_poisson.print(pcout);

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner_pressure_poisson));

    if(preconditioner_pressure_poisson == PreconditionerPressurePoisson::Multigrid)
    {
      multigrid_data_pressure_poisson.print(pcout);
    }
  }

  void
  print_parameters_projection_step(ConditionalOStream & pcout)
  {
    if(use_divergence_penalty == true)
    {
      print_parameter(pcout, "Solver projection step", enum_to_string(solver_projection));

      solver_data_projection.print(pcout);

      if(use_divergence_penalty == true && use_continuity_penalty == true)
      {
        print_parameter(pcout,
                        "Preconditioner projection step",
                        enum_to_string(preconditioner_projection));

        if(preconditioner_projection == PreconditionerProjection::BlockJacobi &&
           implement_block_diagonal_preconditioner_matrix_free)
        {
          print_parameter(pcout,
                          "Preconditioner block diagonal",
                          enum_to_string(preconditioner_block_diagonal_projection));

          solver_data_block_diagonal_projection.print(pcout);
        }

        print_parameter(pcout,
                        "Update preconditioner projection step",
                        update_preconditioner_projection);
      }
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

      newton_solver_data_convective.print(pcout);

      pcout << "  Linear solver:" << std::endl;

      solver_data_convective.print(pcout);
    }

    // projection method
    print_parameters_pressure_poisson(pcout);

    // projection step
    pcout << std::endl << "  Projection step:" << std::endl;
    print_parameters_projection_step(pcout);

    // Viscous step
    pcout << std::endl << "  Viscous step:" << std::endl;

    print_parameter(pcout, "Solver viscous step", enum_to_string(solver_viscous));

    solver_data_viscous.print(pcout);

    print_parameter(pcout, "Preconditioner viscous step", enum_to_string(preconditioner_viscous));

    if(preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      multigrid_data_viscous.print(pcout);
    }

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

      newton_solver_data_momentum.print(pcout);

      pcout << std::endl;
    }

    // Solver linear(ized) problem
    pcout << "  Linear solver:" << std::endl;

    print_parameter(pcout, "Solver", enum_to_string(solver_momentum));

    solver_data_momentum.print(pcout);

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner_momentum));

    if(preconditioner_momentum == MomentumPreconditioner::Multigrid)
    {
      print_parameter(pcout,
                      "Multigrid operator type",
                      enum_to_string(multigrid_operator_type_momentum));

      multigrid_data_momentum.print(pcout);
    }

    print_parameter(pcout, "Update of preconditioner", update_preconditioner_momentum);

    if(update_preconditioner_momentum == true)
    {
      // if a nonlinear problem has to be solved
      if(equation_type == EquationType::NavierStokes &&
         treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      {
        print_parameter(pcout,
                        "Update every Newton iterations",
                        update_preconditioner_momentum_every_newton_iter);
      }

      print_parameter(pcout,
                      "Update every time steps",
                      update_preconditioner_momentum_every_time_steps);
    }

    // formulations of pressure-correction scheme
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

    // Newton solver

    // if a nonlinear problem has to be solved
    if(equation_type == EquationType::NavierStokes &&
       treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    {
      pcout << "Newton solver:" << std::endl;

      newton_solver_data_coupled.print(pcout);

      pcout << std::endl;
    }

    // Solver linearized problem
    pcout << "Linear solver:" << std::endl;

    print_parameter(pcout, "Solver", enum_to_string(solver_coupled));

    solver_data_coupled.print(pcout);

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner_coupled));

    print_parameter(pcout, "Update preconditioner", update_preconditioner_coupled);

    if(update_preconditioner_coupled == true)
    {
      // if a nonlinear problem has to be solved
      if(equation_type == EquationType::NavierStokes &&
         treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      {
        print_parameter(pcout,
                        "Update every Newton iterations",
                        update_preconditioner_coupled_every_newton_iter);
      }

      print_parameter(pcout,
                      "Update every time steps",
                      update_preconditioner_coupled_every_time_steps);
    }

    pcout << std::endl << "  Velocity/momentum block:" << std::endl;

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner_velocity_block));

    if(preconditioner_velocity_block == MomentumPreconditioner::Multigrid)
    {
      print_parameter(pcout,
                      "Multigrid operator type",
                      enum_to_string(multigrid_operator_type_velocity_block));

      multigrid_data_velocity_block.print(pcout);

      print_parameter(pcout,
                      "Exact inversion of velocity block",
                      exact_inversion_of_velocity_block);

      if(exact_inversion_of_velocity_block == true)
      {
        solver_data_velocity_block.print(pcout);
      }
    }

    pcout << std::endl << "  Pressure/Schur-complement block:" << std::endl;

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner_pressure_block));

    if(preconditioner_pressure_block == SchurComplementPreconditioner::LaplaceOperator ||
       preconditioner_pressure_block == SchurComplementPreconditioner::CahouetChabard ||
       preconditioner_pressure_block == SchurComplementPreconditioner::Elman ||
       preconditioner_pressure_block == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      print_parameter(pcout,
                      "Discretization of Laplacian",
                      enum_to_string(discretization_of_laplacian));

      multigrid_data_pressure_block.print(pcout);

      print_parameter(pcout,
                      "Exact inversion of Laplace operator",
                      exact_inversion_of_laplace_operator);

      if(exact_inversion_of_laplace_operator)
      {
        solver_data_pressure_block.print(pcout);
      }
    }

    // projection_step
    if(use_divergence_penalty == true || use_continuity_penalty == true)
    {
      if(add_penalty_terms_to_monolithic_system == false)
      {
        pcout << std::endl << "Postprocessing of velocity:" << std::endl;
        print_parameters_projection_step(pcout);
      }
    }
  }

  void
  print_parameters_turbulence(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Turbulence:" << std::endl;

    print_parameter(pcout, "Use turbulence model", use_turbulence_model);

    if(use_turbulence_model == true)
    {
      print_parameter(pcout, "Turbulence model", enum_to_string(turbulence_model));
      print_parameter(pcout, "Turbulence model constant", turbulence_model_constant);
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
    if(solver_type == SolverType::Unsteady)
    {
      solver_info_data.print(pcout);
    }

    // restart
    if(solver_type == SolverType::Unsteady)
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

  // use adaptive time stepping?
  bool adaptive_time_stepping;

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

  // triangulation type
  TriangulationType triangulation_type;

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

  // solver data for pressure Poisson equation
  SolverData solver_data_pressure_poisson;

  // description: see enum declaration
  PreconditionerPressurePoisson preconditioner_pressure_poisson;

  // update of preconditioner for this equation is currently not provided and not needed

  // description: see declaration of MultigridData
  MultigridData multigrid_data_pressure_poisson;

  // PROJECTION STEP

  // description: see enum declaration
  SolverProjection solver_projection;

  // solver data for projection step
  SolverData solver_data_projection;

  // description: see enum declaration
  PreconditionerProjection preconditioner_projection;

  // Update preconditioner before solving the linear system of equations.
  // Note that this variable is only used when using an iterative method
  // to solve the global projection equation.
  bool update_preconditioner_projection;

  // Update preconditioner every ... time steps.
  // This variable is only used if update of preconditioner is true.
  unsigned int update_preconditioner_projection_every_time_steps;

  // description: see enum declaration (only relevant if block diagonal is used as
  // preconditioner)
  PreconditionerBlockDiagonal preconditioner_block_diagonal_projection;

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

  // CONVECTIVE STEP
  NewtonSolverData newton_solver_data_convective;

  // solver data for linearized problem
  SolverData solver_data_convective;

  // update of preconditioner for this equation is currently not provided

  // VISCOUS STEP

  // description: see enum declaration
  SolverViscous solver_viscous;

  // solver data for viscous step
  SolverData solver_data_viscous;

  // description: see enum declaration
  PreconditionerViscous preconditioner_viscous;

  // update preconditioner before every solve of the viscous step
  bool update_preconditioner_viscous;

  // Update preconditioner every ... time steps.
  // This variable is only used if update of preconditioner is true.
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

  // turbulence parameters that are required for statistics (post-processing)
  TurbulenceStatisticsData turb_stat_data;

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
  SolverInfoData solver_info_data;

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
