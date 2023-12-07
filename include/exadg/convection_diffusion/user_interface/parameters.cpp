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

// ExaDG
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace ConvDiff
{
Parameters::Parameters()
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
    n_refine_time(0),
    cfl(-1.),
    max_velocity(std::numeric_limits<double>::min()),
    diffusion_number(-1.),
    c_eff(-1.),
    exponent_fe_degree_convection(1.5),
    exponent_fe_degree_diffusion(3.0),
    restarted_simulation(false),
    restart_data(RestartData()),

    // SPATIAL DISCRETIZATION
    grid(GridData()),
    mapping_degree(1),
    degree(1),
    enable_adaptivity(false),
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
Parameters::check() const
{
  // MATHEMATICAL MODEL
  AssertThrow(problem_type != ProblemType::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  AssertThrow(equation_type != EquationType::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  if(analytical_velocity_field)
  {
    if(equation_type == EquationType::Convection or
       equation_type == EquationType::ConvectionDiffusion)
    {
      if(temporal_discretization == TemporalDiscretization::ExplRK)
      {
        AssertThrow(store_analytical_velocity_in_dof_vector == false,
                    dealii::ExcMessage(
                      "This option is not implemented for explicit Runge-Kutta time stepping."));
      }
    }
  }

  // moving mesh
  if(ale_formulation)
  {
    AssertThrow(problem_type == ProblemType::Unsteady,
                dealii::ExcMessage("Problem type has to be Unsteady when using ALE formulation."));

    AssertThrow(
      temporal_discretization == TemporalDiscretization::BDF and
        (treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit or
         treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit),
      dealii::ExcMessage(
        "ALE formulation is not implemented for the specified type of time integration."));

    AssertThrow(equation_type == EquationType::Convection or
                  equation_type == EquationType::ConvectionDiffusion,
                dealii::ExcMessage(
                  "ALE formulation can only be used if the problem involves convection."));

    if(analytical_velocity_field)
    {
      AssertThrow(
        store_analytical_velocity_in_dof_vector == true,
        dealii::ExcMessage(
          "When using the ALE formulation, the velocity has to be stored in a DoF vector."));
    }

    AssertThrow(
      formulation_convective_term == FormulationConvectiveTerm::ConvectiveFormulation,
      dealii::ExcMessage(
        "ALE formulation can only be used in combination with convective formulation of convective term."));
  }

  // PHYSICAL QUANTITIES
  AssertThrow(end_time > start_time, dealii::ExcMessage("parameter must be defined"));

  // Set the diffusivity whenever the diffusive term is involved.
  if(equation_type == EquationType::Diffusion or equation_type == EquationType::ConvectionDiffusion)
  {
    AssertThrow(diffusivity > (0.0 - std::numeric_limits<double>::epsilon()),
                dealii::ExcMessage("parameter must be defined"));
  }

  // TEMPORAL DISCRETIZATION
  if(problem_type == ProblemType::Unsteady)
  {
    AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
                dealii::ExcMessage("parameter must be defined"));

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      if(equation_type == EquationType::Convection or
         equation_type == EquationType::ConvectionDiffusion)
      {
        AssertThrow(treatment_of_convective_term != TreatmentOfConvectiveTerm::Undefined,
                    dealii::ExcMessage("parameter must be defined"));
      }
    }

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      AssertThrow(time_integrator_rk != TimeIntegratorRK::Undefined,
                  dealii::ExcMessage("parameter must be defined"));
    }

    AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
                dealii::ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
      AssertThrow(time_step_size > 0.0, dealii::ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
      AssertThrow(c_eff > 0., dealii::ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::CFL)
    {
      AssertThrow(
        equation_type == EquationType::Convection or
          equation_type == EquationType::ConvectionDiffusion,
        dealii::ExcMessage(
          "Type of time step calculation CFL does not make sense for the specified equation type."));

      AssertThrow(cfl > 0., dealii::ExcMessage("parameter must be defined"));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::Diffusion)
    {
      AssertThrow(
        equation_type == EquationType::Diffusion or
          equation_type == EquationType::ConvectionDiffusion,
        dealii::ExcMessage(
          "Type of time step calculation Diffusion does not make sense for the specified equation type."));

      AssertThrow(diffusion_number > 0., dealii::ExcMessage("parameter must be defined"));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
    {
      AssertThrow(
        equation_type == EquationType::ConvectionDiffusion,
        dealii::ExcMessage(
          "Type of time step calculation CFLAndDiffusion does not make sense for the specified equation type."));

      AssertThrow(cfl > 0., dealii::ExcMessage("parameter must be defined"));
      AssertThrow(diffusion_number > 0., dealii::ExcMessage("parameter must be defined"));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::Diffusion or
       calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
    {
      AssertThrow(
        temporal_discretization == TemporalDiscretization::ExplRK,
        dealii::ExcMessage(
          "Diffusion-type time step calculation can only be used in case of explicit RK time integration."));
    }

    if(adaptive_time_stepping == true)
    {
      AssertThrow(calculation_of_time_step_size == TimeStepCalculation::CFL or
                    calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion,
                  dealii::ExcMessage(
                    "Adaptive time stepping can only be used in combination with CFL condition."));
    }

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      AssertThrow(order_time_integrator >= 1 and order_time_integrator <= 4,
                  dealii::ExcMessage("Specified order of time integrator ExplRK not implemented!"));
    }

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      AssertThrow(order_time_integrator >= 1 and order_time_integrator <= 4,
                  dealii::ExcMessage("Specified order of time integrator BDF not implemented!"));
    }
  }

  // SPATIAL DISCRETIZATION
  grid.check();

  if(enable_adaptivity)
  {
    AssertThrow(not ale_formulation,
                dealii::ExcMessage("Combination of adaptive mesh refinement "
                                   "and ALE formulation not implemented."));

    AssertThrow(temporal_discretization == TemporalDiscretization::BDF,
                dealii::ExcMessage("Adaptive mesh refinement only implemented"
                                   "for implicit time integration."));

    AssertThrow(grid.element_type == ElementType::Hypercube,
                dealii::ExcMessage("Adaptive mesh refinement is currently "
                                   "only supported for hypercube elements."));
  }

  AssertThrow(degree > 0, dealii::ExcMessage("Polynomial degree must be larger than zero."));

  if(equation_type == EquationType::Convection or
     equation_type == EquationType::ConvectionDiffusion)
  {
    AssertThrow(numerical_flux_convective_operator != NumericalFluxConvectiveOperator::Undefined,
                dealii::ExcMessage("parameter must be defined"));
  }


  // SOLVER
  if(temporal_discretization == TemporalDiscretization::BDF)
  {
    AssertThrow(solver != Solver::Undefined, dealii::ExcMessage("parameter must be defined"));

    AssertThrow(preconditioner != Preconditioner::Undefined,
                dealii::ExcMessage("parameter must be defined"));

    if(preconditioner == Preconditioner::Multigrid)
    {
      AssertThrow(mg_operator_type != MultigridOperatorType::Undefined,
                  dealii::ExcMessage("parameter must be defined"));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
      {
        AssertThrow(mg_operator_type != MultigridOperatorType::ReactionConvection and
                      mg_operator_type != MultigridOperatorType::ReactionConvectionDiffusion,
                    dealii::ExcMessage(
                      "Invalid solver parameters. The convective term is treated explicitly."));
      }

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      {
        if(equation_type == EquationType::Convection)
        {
          AssertThrow(mg_operator_type == MultigridOperatorType::ReactionConvection,
                      dealii::ExcMessage(
                        "Invalid solver parameters. A purely diffusive problem is considered."));
        }
      }

      if(equation_type == EquationType::Diffusion)
      {
        AssertThrow(mg_operator_type == MultigridOperatorType::ReactionDiffusion,
                    dealii::ExcMessage(
                      "Invalid solver parameters. A purely diffusive problem is considered."));
      }
    }
  }
  else
  {
    AssertThrow(temporal_discretization == TemporalDiscretization::ExplRK,
                dealii::ExcMessage("Not implemented"));
  }

  if(implement_block_diagonal_preconditioner_matrix_free)
  {
    AssertThrow(
      use_cell_based_face_loops == true,
      dealii::ExcMessage(
        "Cell based face loops have to be used for matrix-free implementation of block diagonal preconditioner."));

    AssertThrow(
      solver_block_diagonal != Elementwise::Solver::Undefined,
      dealii::ExcMessage(
        "Invalid parameter. A solver type needs to be specified for elementwise matrix-free iterative solver."));
  }

  // NUMERICAL PARAMETERS
}

bool
Parameters::involves_h_multigrid() const
{
  if(linear_system_has_to_be_solved() and preconditioner == Preconditioner::Multigrid and
     multigrid_data.involves_h_transfer())
    return true;
  else
    return false;
}

bool
Parameters::linear_system_including_convective_term_has_to_be_solved() const
{
  bool equation_with_convective_term =
    equation_type == EquationType::Convection or equation_type == EquationType::ConvectionDiffusion;

  bool solver_with_convective_term =
    problem_type == ProblemType::Steady or
    (problem_type == ProblemType::Unsteady and
     temporal_discretization == TemporalDiscretization::BDF and
     treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit);

  return (equation_with_convective_term and solver_with_convective_term);
}

bool
Parameters::convective_problem() const
{
  return (equation_type == EquationType::Convection or
          equation_type == EquationType::ConvectionDiffusion);
}

bool
Parameters::diffusive_problem() const
{
  return (equation_type == EquationType::Diffusion or
          equation_type == EquationType::ConvectionDiffusion);
}


bool
Parameters::linear_system_has_to_be_solved() const
{
  bool linear_solver_needed =
    problem_type == ProblemType::Steady or (problem_type == ProblemType::Unsteady and
                                            temporal_discretization == TemporalDiscretization::BDF);

  return linear_solver_needed;
}

TypeVelocityField
Parameters::get_type_velocity_field() const
{
  if(analytical_velocity_field == true and store_analytical_velocity_in_dof_vector == false)
    return TypeVelocityField::Function;
  else
    return TypeVelocityField::DoFVector;
}

void
Parameters::print(dealii::ConditionalOStream const & pcout, std::string const & name) const
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
Parameters::print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Problem type", problem_type);
  print_parameter(pcout, "Equation type", equation_type);
  print_parameter(pcout, "Right-hand side", right_hand_side);

  if(equation_type == EquationType::Convection or
     equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "Analytical velocity field", analytical_velocity_field);
    print_parameter(pcout, "ALE formulation", ale_formulation);
    print_parameter(pcout, "Formulation convective term", formulation_convective_term);
  }
}

void
Parameters::print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Physical quantities:" << std::endl;

  if(problem_type == ProblemType::Unsteady)
  {
    print_parameter(pcout, "Start time", start_time);
    print_parameter(pcout, "End time", end_time);
  }

  // diffusivity
  if(equation_type == EquationType::Diffusion or equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "Diffusivity", diffusivity);
  }
}

void
Parameters::print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  print_parameter(pcout, "Temporal discretization method", temporal_discretization);

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    print_parameter(pcout, "Explicit time integrator", time_integrator_rk);
  }

  print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);

  print_parameter(pcout, "Temporal refinements", n_refine_time);

  if(temporal_discretization == TemporalDiscretization::BDF)
  {
    print_parameter(pcout, "Order of time integrator", order_time_integrator);
    print_parameter(pcout, "Start with low order method", start_with_low_order);
    print_parameter(pcout, "Treatment of convective term", treatment_of_convective_term);
  }

  print_parameter(pcout, "Calculation of time step size", calculation_of_time_step_size);

  print_parameter(pcout, "Adaptive time stepping", adaptive_time_stepping);

  if(adaptive_time_stepping)
  {
    print_parameter(pcout,
                    "Adaptive time stepping limiting factor",
                    adaptive_time_stepping_limiting_factor);

    print_parameter(pcout, "Maximum allowable time step size", time_step_size_max);

    print_parameter(pcout, "Type of CFL condition", adaptive_time_stepping_cfl_type);
  }


  // here we do not print quantities such as cfl, diffusion_number, time_step_size
  // because this is done by the time integration scheme (or the functions that
  // calculate the time step size)

  print_parameter(pcout, "Restarted simulation", restarted_simulation);

  restart_data.print(pcout);
}

void
Parameters::print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  grid.print(pcout);

  print_parameter(pcout, "Mapping degree", mapping_degree);

  print_parameter(pcout, "Polynomial degree", degree);

  if(enable_adaptivity)
  {
    amr_data.print(pcout);
  }

  if(equation_type == EquationType::Convection or
     equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "Numerical flux convective term", numerical_flux_convective_operator);
  }

  if(equation_type == EquationType::Diffusion or equation_type == EquationType::ConvectionDiffusion)
  {
    print_parameter(pcout, "IP factor viscous term", IP_factor);
  }
}

void
Parameters::print_parameters_solver(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Solver:" << std::endl;

  print_parameter(pcout, "Solver", solver);

  solver_data.print(pcout);

  print_parameter(pcout, "Preconditioner", preconditioner);

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
    print_parameter(pcout, "Solver block diagonal", solver_block_diagonal);

    print_parameter(pcout, "Preconditioner block diagonal", preconditioner_block_diagonal);

    solver_data_block_diagonal.print(pcout);
  }

  if(preconditioner == Preconditioner::Multigrid)
  {
    print_parameter(pcout, "Multigrid operator type", mg_operator_type);
    multigrid_data.print(pcout);
  }

  solver_info_data.print(pcout);
}


void
Parameters::print_parameters_numerical_parameters(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Use cell-based face loops", use_cell_based_face_loops);

  if(temporal_discretization == TemporalDiscretization::ExplRK)
    print_parameter(pcout, "Use combined operator", use_combined_operator);

  if(analytical_velocity_field)
  {
    if(temporal_discretization == TemporalDiscretization::BDF and
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
} // namespace ExaDG
