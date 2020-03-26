/*
 * input_parameters.cpp
 *
 *  Created on: May 15, 2019
 *      Author: fehn
 */

#include <deal.II/base/exceptions.h>

#include "input_parameters.h"

namespace CompNS
{
// standard constructor that initializes parameters with default values
InputParameters::InputParameters()
  : // MATHEMATICAL MODEL
    equation_type(EquationType::Undefined),
    right_hand_side(false),

    // PHYSICAL QUANTITIES
    start_time(0.),
    end_time(-1.),
    dynamic_viscosity(0.),
    reference_density(1.0),
    heat_capacity_ratio(1.4),
    thermal_conductivity(0.0262),
    specific_gas_constant(287.058),
    max_temperature(273.15),

    // TEMPORAL DISCRETIZATION
    temporal_discretization(TemporalDiscretization::Undefined),
    order_time_integrator(1),
    stages(1),
    calculation_of_time_step_size(TimeStepCalculation::Undefined),
    time_step_size(-1.),
    max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
    max_velocity(-1.),
    cfl_number(-1.),
    diffusion_number(-1.),
    exponent_fe_degree_cfl(2.0),
    exponent_fe_degree_viscous(4.0),
    dt_refinements(0),
    // restart
    restarted_simulation(false),
    restart_data(RestartData()),
    solver_info_data(SolverInfoData()),

    // SPATIAL DISCRETIZATION

    // triangulation
    triangulation_type(TriangulationType::Undefined),
    degree(3),
    mapping(MappingType::Affine),
    n_q_points_convective(QuadratureRule::Standard),
    n_q_points_viscous(QuadratureRule::Standard),
    h_refinements(0),

    // viscous term
    IP_factor(1.0),

    // NUMERICAL PARAMETERS
    detect_instabilities(true),
    use_combined_operator(false)
{
}

void
InputParameters::check_input_parameters()
{
  // MATHEMATICAL MODEL
  AssertThrow(equation_type != EquationType::Undefined, ExcMessage("parameter must be defined"));


  // PHYSICAL QUANTITIES
  AssertThrow(end_time > start_time, ExcMessage("parameter must be defined"));


  // TEMPORAL DISCRETIZATION
  AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
              ExcMessage("parameter must be defined"));

  AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
              ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
    AssertThrow(time_step_size > 0.0, ExcMessage("parameter must be defined"));

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                ExcMessage("Specified order of time integrator ExplRK not implemented!"));
  }

  if(temporal_discretization == TemporalDiscretization::SSPRK)
  {
    AssertThrow(stages >= 1, ExcMessage("Specify number of RK stages!"));
  }

  if(calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    AssertThrow(max_velocity >= 0.0, ExcMessage("Invalid parameter max_velocity."));
    AssertThrow(cfl_number > 0.0, ExcMessage("parameter must be defined"));
    AssertThrow(diffusion_number > 0.0, ExcMessage("parameter must be defined"));
  }


  // SPATIAL DISCRETIZATION
  AssertThrow(triangulation_type != TriangulationType::Undefined,
              ExcMessage("parameter must be defined"));

  AssertThrow(degree > 0, ExcMessage("Invalid parameter."));

  if(use_combined_operator)
  {
    AssertThrow(
      n_q_points_convective == n_q_points_viscous,
      ExcMessage(
        "For the combined operator, both convective and viscous terms have to be integrated with the same number of quadrature points."));
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
  // If a system of equations has to be solved (currently not used)
  print_parameters_solver(pcout);

  // NUMERICAL PARAMETERS
  print_parameters_numerical_parameters(pcout);
}

void
InputParameters::print_parameters_mathematical_model(ConditionalOStream & pcout)
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Equation type", enum_to_string(equation_type));
  print_parameter(pcout, "Right-hand side", right_hand_side);
}

void
InputParameters::print_parameters_physical_quantities(ConditionalOStream & pcout)
{
  pcout << std::endl << "Physical quantities:" << std::endl;

  print_parameter(pcout, "Start time", start_time);
  print_parameter(pcout, "End time", end_time);

  print_parameter(pcout, "Dynamic viscosity", dynamic_viscosity);
  print_parameter(pcout, "Reference density", reference_density);
  print_parameter(pcout, "Heat capacity ratio", heat_capacity_ratio);
  print_parameter(pcout, "Thermal conductivity", thermal_conductivity);
  print_parameter(pcout, "Specific gas constant", specific_gas_constant);
}

void
InputParameters::print_parameters_temporal_discretization(ConditionalOStream & pcout)
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  print_parameter(pcout, "Temporal discretization method", enum_to_string(temporal_discretization));

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    print_parameter(pcout, "Order of time integrator", order_time_integrator);
  }

  if(temporal_discretization == TemporalDiscretization::SSPRK)
  {
    print_parameter(pcout, "Order of time integrator", order_time_integrator);
    print_parameter(pcout, "Number of stages", stages);
  }

  print_parameter(pcout,
                  "Calculation of time step size",
                  enum_to_string(calculation_of_time_step_size));

  // maximum number of time steps
  print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);


  // here we do not print quantities such as cfl_number, diffusion_number, time_step_size
  // because this is done by the time integration scheme (or the functions that
  // calculate the time step size)

  print_parameter(pcout, "Refinement steps dt", dt_refinements);

  print_parameter(pcout, "Restarted simulation", restarted_simulation);

  restart_data.print(pcout);

  solver_info_data.print(pcout);
}

void
InputParameters::print_parameters_spatial_discretization(ConditionalOStream & pcout)
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));

  print_parameter(pcout, "Polynomial degree of shape functions", degree);

  print_parameter(pcout, "Mapping", enum_to_string(mapping));

  print_parameter(pcout, "Quadrature rule convective term", enum_to_string(n_q_points_convective));
  print_parameter(pcout, "Quadrature rule viscous term", enum_to_string(n_q_points_viscous));

  print_parameter(pcout, "Number of h-refinements", h_refinements);

  print_parameter(pcout, "IP factor viscous term", IP_factor);
}

void
InputParameters::print_parameters_solver(ConditionalOStream & /*pcout*/)
{
  /*
  pcout << std::endl
        << "Solver:" << std::endl;
  */
}

void
InputParameters::print_parameters_numerical_parameters(ConditionalOStream & pcout)
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Detect instabilities", detect_instabilities);
  print_parameter(pcout, "Use combined operator", use_combined_operator);
}

} // namespace CompNS
