/*
 * input_parameters.cpp
 *
 *  Created on: May 13, 2019
 *      Author: fehn
 */

#include "input_parameters.h"

#include "../../utilities/print_functions.h"

namespace ConvDiff
{
InputParameters::InputParameters()
  : // MATHEMATICAL MODEL
    problem_type(ProblemType::Undefined),
    equation_type(EquationType::Undefined),
    analytical_velocity_field(true),
    ale_formulation(false),
    right_hand_side(false),
    formulation_convective_term(FormulationConvectiveTerm::DivergenceFormulation),

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
    restarted_simulation(false),
    restart_data(RestartData()),

    // SPATIAL DISCRETIZATION
    triangulation_type(TriangulationType::Undefined),
    mapping(MappingType::Affine),
    numerical_flux_convective_operator(NumericalFluxConvectiveOperator::Undefined),
    IP_factor(1.0),

    // SOLVER
    solver(Solver::Undefined),
    solver_data(SolverData(1e4, 1.e-12, 1.e-6, 100)),
    preconditioner(Preconditioner::Undefined),
    update_preconditioner(false),
    update_preconditioner_every_time_steps(1),
    implement_block_diagonal_preconditioner_matrix_free(false),
    solver_block_diagonal(Elementwise::Solver::Undefined),
    preconditioner_block_diagonal(Elementwise::Preconditioner::InverseMassMatrix),
    solver_data_block_diagonal(SolverData(1000, 1.e-12, 1.e-2, 1000)),
    mg_operator_type(MultigridOperatorType::Undefined),
    multigrid_data(MultigridData()),
    solver_info_data(SolverInfoData()),

    // NUMERICAL PARAMETERS
    use_cell_based_face_loops(false),
    use_combined_operator(true),
    store_analytical_velocity_in_dof_vector(false),
    use_overintegration(false)
{
}

void
InputParameters::check_input_parameters()
{
  // MATHEMATICAL MODEL
  AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("parameter must be defined"));

  AssertThrow(equation_type != EquationType::Undefined, ExcMessage("parameter must be defined"));

  if(analytical_velocity_field)
  {
    if(equation_type == EquationType::Convection ||
       equation_type == EquationType::ConvectionDiffusion)
    {
      if(temporal_discretization == TemporalDiscretization::ExplRK ||
         (temporal_discretization == TemporalDiscretization::BDF &&
          treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF))
      {
        AssertThrow(
          store_analytical_velocity_in_dof_vector == false,
          ExcMessage(
            "This option is not implemented for explicit Runge-Kutta time stepping or BDF time stepping with OIF substepping."));
      }
    }
  }

  // moving mesh
  if(ale_formulation)
  {
    AssertThrow(
      temporal_discretization == TemporalDiscretization::BDF &&
        (treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit ||
         treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit),
      ExcMessage("ALE formulation is not implemented for the specified type of time integration."));

    AssertThrow(equation_type == EquationType::Convection ||
                  equation_type == EquationType::ConvectionDiffusion,
                ExcMessage("ALE formulation can only be used if the problem involves convection."));

    if(analytical_velocity_field)
    {
      AssertThrow(
        store_analytical_velocity_in_dof_vector == true,
        ExcMessage(
          "When using the ALE formulation, the velocity has to be stored in a DoF vector."));
    }

    AssertThrow(
      formulation_convective_term == FormulationConvectiveTerm::ConvectiveFormulation,
      ExcMessage(
        "ALE formulation can only be used in combination with convective formulation of convective term."));
  }

  // PHYSICAL QUANTITIES
  AssertThrow(end_time > start_time, ExcMessage("parameter must be defined"));

  // Set the diffusivity whenever the diffusive term is involved.
  if(equation_type == EquationType::Diffusion || equation_type == EquationType::ConvectionDiffusion)
  {
    AssertThrow(diffusivity > (0.0 + 1.0e-12), ExcMessage("parameter must be defined"));
  }

  // TEMPORAL DISCRETIZATION
  if(problem_type == ProblemType::Unsteady)
  {
    AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
                ExcMessage("parameter must be defined"));

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      if(equation_type == EquationType::Convection ||
         equation_type == EquationType::ConvectionDiffusion)
      {
        AssertThrow(treatment_of_convective_term != TreatmentOfConvectiveTerm::Undefined,
                    ExcMessage("parameter must be defined"));
      }
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
  }

  // SPATIAL DISCRETIZATION
  AssertThrow(triangulation_type != TriangulationType::Undefined,
              ExcMessage("parameter must be defined"));

  if(equation_type == EquationType::Convection ||
     equation_type == EquationType::ConvectionDiffusion)
  {
    AssertThrow(numerical_flux_convective_operator != NumericalFluxConvectiveOperator::Undefined,
                ExcMessage("parameter must be defined"));
  }


  // SOLVER
  if(temporal_discretization == TemporalDiscretization::BDF)
  {
    AssertThrow(solver != Solver::Undefined, ExcMessage("parameter must be defined"));

    AssertThrow(preconditioner != Preconditioner::Undefined,
                ExcMessage("parameter must be defined"));

    if(preconditioner == Preconditioner::Multigrid)
    {
      AssertThrow(mg_operator_type != MultigridOperatorType::Undefined,
                  ExcMessage("parameter must be defined"));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
         treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
      {
        AssertThrow(mg_operator_type != MultigridOperatorType::ReactionConvection &&
                      mg_operator_type != MultigridOperatorType::ReactionConvectionDiffusion,
                    ExcMessage(
                      "Invalid solver parameters. The convective term is treated explicitly."));
      }

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      {
        if(equation_type == EquationType::Convection)
        {
          AssertThrow(mg_operator_type == MultigridOperatorType::ReactionConvection,
                      ExcMessage(
                        "Invalid solver parameters. A purely diffusive problem is considered."));
        }
      }

      if(equation_type == EquationType::Diffusion)
      {
        AssertThrow(mg_operator_type == MultigridOperatorType::ReactionDiffusion,
                    ExcMessage(
                      "Invalid solver parameters. A purely diffusive problem is considered."));
      }
    }
  }
  else
  {
    AssertThrow(temporal_discretization == TemporalDiscretization::ExplRK,
                ExcMessage("Not implemented"));
  }

  if(implement_block_diagonal_preconditioner_matrix_free)
  {
    AssertThrow(
      use_cell_based_face_loops == true,
      ExcMessage(
        "Cell based face loops have to be used for matrix-free implementation of block diagonal preconditioner."));

    AssertThrow(
      solver_block_diagonal != Elementwise::Solver::Undefined,
      ExcMessage(
        "Invalid parameter. A solver type needs to be specified for elementwise matrix-free iterative solver."));
  }

  // NUMERICAL PARAMETERS
}

bool
InputParameters::linear_system_including_convective_term_has_to_be_solved() const
{
  bool equation_with_convective_term =
    equation_type == EquationType::Convection || equation_type == EquationType::ConvectionDiffusion;

  bool solver_with_convective_term =
    problem_type == ProblemType::Steady ||
    (problem_type == ProblemType::Unsteady &&
     temporal_discretization == TemporalDiscretization::BDF &&
     treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit);

  return (equation_with_convective_term && solver_with_convective_term);
}

bool
InputParameters::convective_problem() const
{
  return (equation_type == EquationType::Convection ||
          equation_type == EquationType::ConvectionDiffusion);
}

bool
InputParameters::diffusive_problem() const
{
  return (equation_type == EquationType::Diffusion ||
          equation_type == EquationType::ConvectionDiffusion);
}


bool
InputParameters::linear_system_has_to_be_solved() const
{
  bool linear_solver_needed =
    problem_type == ProblemType::Steady || (problem_type == ProblemType::Unsteady &&
                                            temporal_discretization == TemporalDiscretization::BDF);

  return linear_solver_needed;
}

TypeVelocityField
InputParameters::get_type_velocity_field() const
{
  if(analytical_velocity_field == true && store_analytical_velocity_in_dof_vector == false)
    return TypeVelocityField::Function;
  else
    return TypeVelocityField::DoFVector;
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

  print_parameter(pcout, "Problem type", enum_to_string(problem_type));
  print_parameter(pcout, "Equation type", enum_to_string(equation_type));
  print_parameter(pcout, "Right-hand side", right_hand_side);

  if(equation_type == EquationType::Convection ||
     equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "Analytical velocity field", analytical_velocity_field);
    print_parameter(pcout, "ALE formulation", ale_formulation);
    print_parameter(pcout,
                    "Formulation convective term",
                    enum_to_string(formulation_convective_term));
  }
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

  print_parameter(pcout, "Restarted simulation", restarted_simulation);

  restart_data.print(pcout);
}

void
InputParameters::print_parameters_spatial_discretization(ConditionalOStream & pcout)
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));

  print_parameter(pcout, "Mapping", enum_to_string(mapping));

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
    print_parameter(pcout, "Solver block diagonal", enum_to_string(solver_block_diagonal));

    print_parameter(pcout,
                    "Preconditioner block diagonal",
                    enum_to_string(preconditioner_block_diagonal));

    solver_data_block_diagonal.print(pcout);
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

  if(temporal_discretization == TemporalDiscretization::ExplRK)
    print_parameter(pcout, "Use combined operator", use_combined_operator);

  if(analytical_velocity_field)
  {
    if(temporal_discretization == TemporalDiscretization::BDF &&
       treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    {
      print_parameter(pcout,
                      "Store velocity in DoF vector",
                      store_analytical_velocity_in_dof_vector);
    }
  }

  print_parameter(pcout, "Use over-integration", use_overintegration);
}

} // namespace ConvDiff
