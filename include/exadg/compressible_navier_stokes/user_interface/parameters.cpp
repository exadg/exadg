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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/compressible_navier_stokes/user_interface/parameters.h>

namespace ExaDG
{
namespace CompNS
{
// standard constructor that initializes parameters with default values
Parameters::Parameters()
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
    n_refine_time(0),
    max_velocity(-1.),
    cfl_number(-1.),
    diffusion_number(-1.),
    exponent_fe_degree_cfl(2.0),
    exponent_fe_degree_viscous(4.0),
    // restart
    restarted_simulation(false),
    restart_data(RestartData()),
    solver_info_data(SolverInfoData()),

    // SPATIAL DISCRETIZATION

    // triangulation
    grid(GridData()),
    degree(1),
    n_q_points_convective(QuadratureRule::Standard),
    n_q_points_viscous(QuadratureRule::Standard),

    // viscous term
    IP_factor(1.0),

    // NUMERICAL PARAMETERS
    detect_instabilities(true),
    use_combined_operator(false)
{
}

void
Parameters::check() const
{
  // MATHEMATICAL MODEL
  AssertThrow(equation_type != EquationType::Undefined,
              dealii::ExcMessage("parameter must be defined"));


  // PHYSICAL QUANTITIES
  AssertThrow(end_time > start_time, dealii::ExcMessage("parameter must be defined"));


  // TEMPORAL DISCRETIZATION
  AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
    AssertThrow(time_step_size > 0.0, dealii::ExcMessage("parameter must be defined"));

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    AssertThrow(order_time_integrator >= 1 and order_time_integrator <= 4,
                dealii::ExcMessage("Specified order of time integrator ExplRK not implemented!"));
  }

  if(temporal_discretization == TemporalDiscretization::SSPRK)
  {
    AssertThrow(stages >= 1, dealii::ExcMessage("Specify number of RK stages!"));
  }

  if(calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    AssertThrow(max_velocity >= 0.0, dealii::ExcMessage("Invalid parameter max_velocity."));
    AssertThrow(cfl_number > 0.0, dealii::ExcMessage("parameter must be defined"));
    AssertThrow(diffusion_number > 0.0, dealii::ExcMessage("parameter must be defined"));
  }


  // SPATIAL DISCRETIZATION
  grid.check();

  AssertThrow(degree > 0, dealii::ExcMessage("Polynomial degree must be larger than zero."));

  if(use_combined_operator)
  {
    AssertThrow(
      n_q_points_convective == n_q_points_viscous,
      dealii::ExcMessage(
        "For the combined operator, both convective and viscous terms have to be integrated with the same number of quadrature points."));
  }

  // NUMERICAL PARAMETERS
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
  // If a system of equations has to be solved (currently not used)
  print_parameters_solver(pcout);

  // NUMERICAL PARAMETERS
  print_parameters_numerical_parameters(pcout);
}

void
Parameters::print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Equation type", equation_type);
  print_parameter(pcout, "Right-hand side", right_hand_side);
}

void
Parameters::print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const
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
Parameters::print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  print_parameter(pcout, "Temporal discretization method", temporal_discretization);

  if(temporal_discretization == TemporalDiscretization::ExplRK)
  {
    print_parameter(pcout, "Order of time integrator", order_time_integrator);
  }

  if(temporal_discretization == TemporalDiscretization::SSPRK)
  {
    print_parameter(pcout, "Order of time integrator", order_time_integrator);
    print_parameter(pcout, "Number of stages", stages);
  }

  print_parameter(pcout, "Calculation of time step size", calculation_of_time_step_size);

  // maximum number of time steps
  print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);

  print_parameter(pcout, "Temporal refinements", n_refine_time);


  // here we do not print quantities such as cfl_number, diffusion_number, time_step_size
  // because this is done by the time integration scheme (or the functions that
  // calculate the time step size)

  print_parameter(pcout, "Restarted simulation", restarted_simulation);

  restart_data.print(pcout);

  solver_info_data.print(pcout);
}

void
Parameters::print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  grid.print(pcout);

  print_parameter(pcout, "Polynomial degree", degree);

  print_parameter(pcout, "Quadrature rule convective term", n_q_points_convective);
  print_parameter(pcout, "Quadrature rule viscous term", n_q_points_viscous);

  print_parameter(pcout, "IP factor viscous term", IP_factor);
}

void
Parameters::print_parameters_solver(dealii::ConditionalOStream const & /*pcout*/) const
{
  /*
  pcout << std::endl
        << "Solver:" << std::endl;
  */
}

void
Parameters::print_parameters_numerical_parameters(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Detect instabilities", detect_instabilities);
  print_parameter(pcout, "Use combined operator", use_combined_operator);
}

} // namespace CompNS
} // namespace ExaDG
