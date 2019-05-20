/*
 * input_parameters.cpp
 *
 *  Created on: May 13, 2019
 *      Author: fehn
 */

#include "input_parameters.h"

#include "../../functionalities/print_functions.h"

namespace ConvDiff
{
InputParameters::InputParameters()
  : // MATHEMATICAL MODEL
    dim(2),
    problem_type(ProblemType::Undefined),
    equation_type(EquationType::Undefined),
    type_velocity_field(TypeVelocityField::Analytical),
    right_hand_side(false),

    // PHYSICAL QUANTITIES
    start_time(0.),
    end_time(-1.),
    diffusivity(0.),

    // TEMPORAL DISCRETIZATION
    temporal_discretization(TemporalDiscretization::Undefined),
    time_integrator_rk(TimeIntegratorRK::Undefined),
    order_time_integrator(1),
    start_with_low_order(true),
    treatment_of_convective_term(TreatmentOfConvectiveTerm::Undefined),
    calculation_of_time_step_size(TimeStepCalculation::Undefined),
    adaptive_time_stepping(false),
    adaptive_time_stepping_limiting_factor(1.2),
    time_step_size_max(std::numeric_limits<double>::max()),
    adaptive_time_stepping_cfl_type(CFLConditionType::VelocityNorm),
    time_step_size(-1.),
    max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
    cfl(-1.),
    max_velocity(std::numeric_limits<double>::min()),
    time_integrator_oif(TimeIntegratorRK::Undefined),
    cfl_oif(-1.),
    diffusion_number(-1.),
    c_eff(-1.),
    exponent_fe_degree_convection(1.5),
    exponent_fe_degree_diffusion(3.0),
    dt_refinements(0),
    restarted_simulation(false),
    restart_data(RestartData()),

    // SPATIAL DISCRETIZATION
    triangulation_type(TriangulationType::Undefined),
    degree(1),
    mapping(MappingType::Affine),
    h_refinements(0),
    numerical_flux_convective_operator(NumericalFluxConvectiveOperator::Undefined),
    IP_factor(1.0),

    // SOLVER
    solver(Solver::Undefined),
    solver_data(SolverData(1e4, 1.e-12, 1.e-6, 100)),
    preconditioner(Preconditioner::Undefined),
    update_preconditioner(false),
    update_preconditioner_every_time_steps(1),
    implement_block_diagonal_preconditioner_matrix_free(false),
    preconditioner_block_diagonal(PreconditionerBlockDiagonal::InverseMassMatrix),
    block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2, 1000)),
    mg_operator_type(MultigridOperatorType::Undefined),
    multigrid_data(MultigridData()),
    solver_info_data(SolverInfoData()),

    // NUMERICAL PARAMETERS
    use_cell_based_face_loops(false),
    runtime_optimization(false)
{
}

void
InputParameters::check_input_parameters()
{
  // MATHEMATICAL MODEL
  AssertThrow(dim == 2 || dim == 3, ExcMessage("Invalid parameter."));

  AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("parameter must be defined"));

  AssertThrow(equation_type != EquationType::Undefined, ExcMessage("parameter must be defined"));

  // PHYSICAL QUANTITIES
  AssertThrow(end_time > start_time, ExcMessage("parameter must be defined"));

  // Set the diffusivity whenever the diffusive term is involved.
  if(equation_type == EquationType::Diffusion || equation_type == EquationType::ConvectionDiffusion)
    AssertThrow(diffusivity > (0.0 + 1.0e-12), ExcMessage("parameter must be defined"));


  // TEMPORAL DISCRETIZATION
  AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
              ExcMessage("parameter must be defined"));

  if(temporal_discretization == TemporalDiscretization::BDF)
  {
    AssertThrow(treatment_of_convective_term != TreatmentOfConvectiveTerm::Undefined,
                ExcMessage("parameter must be defined"));
  }

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    AssertThrow(time_integrator_rk != TimeIntegratorRK::Undefined,
                ExcMessage("parameter must be defined"));
  }

  AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
              ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
    AssertThrow(time_step_size > 0.0, ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
    AssertThrow(c_eff > 0., ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    AssertThrow(
      equation_type == EquationType::Convection ||
        equation_type == EquationType::ConvectionDiffusion,
      ExcMessage(
        "Type of time step calculation CFL does not make sense for the specified equation type."));
  }

  if(calculation_of_time_step_size == TimeStepCalculation::Diffusion)
  {
    AssertThrow(
      equation_type == EquationType::Diffusion ||
        equation_type == EquationType::ConvectionDiffusion,
      ExcMessage(
        "Type of time step calculation Diffusion does not make sense for the specified equation type."));
  }

  if(calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    AssertThrow(
      equation_type == EquationType::ConvectionDiffusion,
      ExcMessage(
        "Type of time step calculation CFLAndDiffusion does not make sense for the specified equation type."));
  }

  if(adaptive_time_stepping == true)
  {
    AssertThrow(calculation_of_time_step_size == TimeStepCalculation::CFL ||
                  calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion,
                ExcMessage(
                  "Adaptive time stepping can only be used in combination with CFL condition."));
  }

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                ExcMessage("Specified order of time integrator ExplRK not implemented!"));

    // for the explicit RK method both the convective and the diffusive term are
    // treated explicitly -> one has to specify both the CFL-number and the Diffusion-number
    AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
    AssertThrow(diffusion_number > 0., ExcMessage("parameter must be defined"));
  }

  if(temporal_discretization == TemporalDiscretization::BDF)
  {
    AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                ExcMessage("Specified order of time integrator BDF not implemented!"));

    if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      AssertThrow(time_integrator_oif != TimeIntegratorRK::Undefined,
                  ExcMessage("parameter must be defined"));

      AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
      AssertThrow(cfl_oif > 0., ExcMessage("parameter must be defined"));
    }
  }

  // SPATIAL DISCRETIZATION
  AssertThrow(triangulation_type != TriangulationType::Undefined,
              ExcMessage("parameter must be defined"));

  AssertThrow(degree > 0, ExcMessage("Invalid parameter."));

  if(equation_type == EquationType::Convection ||
     equation_type == EquationType::ConvectionDiffusion || runtime_optimization == true)
  {
    AssertThrow(numerical_flux_convective_operator != NumericalFluxConvectiveOperator::Undefined,
                ExcMessage("parameter must be defined"));
  }


  // SOLVER
  if(temporal_discretization != TemporalDiscretization::ExplRK)
  {
    AssertThrow(solver != Solver::Undefined, ExcMessage("parameter must be defined"));

    AssertThrow(preconditioner != Preconditioner::Undefined,
                ExcMessage("parameter must be defined"));

    if(preconditioner == Preconditioner::Multigrid)
    {
      AssertThrow(mg_operator_type != MultigridOperatorType::Undefined,
                  ExcMessage("parameter must be defined"));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
      {
        AssertThrow(mg_operator_type != MultigridOperatorType::ReactionConvection &&
                      mg_operator_type != MultigridOperatorType::ReactionConvectionDiffusion,
                    ExcMessage(
                      "Invalid solver parameters. The convective term is treated explicitly."));
      }
    }
  }

  if(implement_block_diagonal_preconditioner_matrix_free)
  {
    AssertThrow(
      use_cell_based_face_loops == true,
      ExcMessage(
        "Cell based face loops have to be used for matrix-free implementation of block diagonal preconditioner."));
  }

  // NUMERICAL PARAMETERS
}

void
InputParameters::print(ConditionalOStream & pcout, std::string const & name)
{
  pcout << std::endl << name << std::endl;

  // MATHEMATICAL MODEL
  print_parameters_mathematical_model(pcout);

  // PHYSICAL QUANTITIES
  print_parameters_physical_quantities(pcout);

  // TEMPORAL DISCRETIZATION
  print_parameters_temporal_discretization(pcout);

  // SPATIAL DISCRETIZATION
  print_parameters_spatial_discretization(pcout);

  // SOLVER
  // If a linear system of equations has to be solved:
  // for the currently implementation this means
  if(temporal_discretization != TemporalDiscretization::ExplRK)
    print_parameters_solver(pcout);

  // NUMERICAL PARAMETERS
  print_parameters_numerical_parameters(pcout);
}

void
InputParameters::print_parameters_mathematical_model(ConditionalOStream & pcout)
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Space dimensions", dim);
  print_parameter(pcout, "Problem type", enum_to_string(problem_type));
  print_parameter(pcout, "Equation type", enum_to_string(equation_type));
  print_parameter(pcout, "Type of velocity field", enum_to_string(type_velocity_field));
  print_parameter(pcout, "Right-hand side", right_hand_side);
}

void
InputParameters::print_parameters_physical_quantities(ConditionalOStream & pcout)
{
  pcout << std::endl << "Physical quantities:" << std::endl;

  if(problem_type == ProblemType::Unsteady)
  {
    print_parameter(pcout, "Start time", start_time);
    print_parameter(pcout, "End time", end_time);
  }

  // diffusivity
  if(equation_type == EquationType::Diffusion || equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "Diffusivity", diffusivity);
  }
}

void
InputParameters::print_parameters_temporal_discretization(ConditionalOStream & pcout)
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  print_parameter(pcout, "Temporal discretization method", enum_to_string(temporal_discretization));

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    print_parameter(pcout, "Explicit time integrator", enum_to_string(time_integrator_rk));
  }

  // maximum number of time steps
  print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);

  if(temporal_discretization == TemporalDiscretization::BDF)
  {
    print_parameter(pcout, "Order of time integrator", order_time_integrator);
    print_parameter(pcout, "Start with low order method", start_with_low_order);
    print_parameter(pcout,
                    "Treatment of convective term",
                    enum_to_string(treatment_of_convective_term));

    if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      print_parameter(pcout,
                      "Time integrator for OIF splitting",
                      enum_to_string(time_integrator_oif));
    }
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

    print_parameter(pcout, "Maximum allowable time step size", time_step_size_max);

    print_parameter(pcout,
                    "Type of CFL condition",
                    enum_to_string(adaptive_time_stepping_cfl_type));
  }


  // here we do not print quantities such as cfl, diffusion_number, time_step_size
  // because this is done by the time integration scheme (or the functions that
  // calculate the time step size)

  print_parameter(pcout, "Refinement steps dt", dt_refinements);

  print_parameter(pcout, "Restarted simulation", restarted_simulation);

  restart_data.print(pcout);
}

void
InputParameters::print_parameters_spatial_discretization(ConditionalOStream & pcout)
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));

  print_parameter(pcout, "Polynomial degree of shape functions", degree);

  print_parameter(pcout, "Mapping", enum_to_string(mapping));

  print_parameter(pcout, "Number of h-refinements", h_refinements);

  if(equation_type == EquationType::Convection ||
     equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout,
                    "Numerical flux convective term",
                    enum_to_string(numerical_flux_convective_operator));
  }

  if(equation_type == EquationType::Diffusion || equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "IP factor viscous term", IP_factor);
  }
}

void
InputParameters::print_parameters_solver(ConditionalOStream & pcout)
{
  pcout << std::endl << "Solver:" << std::endl;

  print_parameter(pcout, "Solver", enum_to_string(solver));

  solver_data.print(pcout);

  print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner));

  if(preconditioner != Preconditioner::None)
  {
    print_parameter(pcout, "Update preconditioner", update_preconditioner);

    if(update_preconditioner)
      print_parameter(pcout, "Update every time steps", update_preconditioner_every_time_steps);
  }

  print_parameter(pcout,
                  "Block Jacobi matrix-free",
                  implement_block_diagonal_preconditioner_matrix_free);

  if(implement_block_diagonal_preconditioner_matrix_free)
  {
    print_parameter(pcout,
                    "Preconditioner block diagonal",
                    enum_to_string(preconditioner_block_diagonal));

    block_jacobi_solver_data.print(pcout);
  }

  if(preconditioner == Preconditioner::Multigrid)
  {
    print_parameter(pcout, "Multigrid operator type", enum_to_string(mg_operator_type));
    multigrid_data.print(pcout);
  }

  solver_info_data.print(pcout);
}


void
InputParameters::print_parameters_numerical_parameters(ConditionalOStream & pcout)
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Use cell-based face loops", use_cell_based_face_loops);
  print_parameter(pcout,
                  "Block Jacobi implemented matrix-free",
                  implement_block_diagonal_preconditioner_matrix_free);
  print_parameter(pcout, "Runtime optimization", runtime_optimization);
}

} // namespace ConvDiff
