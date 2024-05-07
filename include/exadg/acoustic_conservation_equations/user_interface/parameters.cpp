/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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
#include <exadg/acoustic_conservation_equations/user_interface/parameters.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Acoustics
{
// standard constructor that initializes parameters
Parameters::Parameters()
  : // MATHEMATICAL MODEL
    formulation(Formulation::Undefined),
    right_hand_side(false),
    aero_acoustic_source_term(false),

    // PHYSICAL QUANTITIES
    start_time(0.),
    end_time(-1.),
    speed_of_sound(-1.),

    // TEMPORAL DISCRETIZATION
    calculation_of_time_step_size(TimeStepCalculation::Undefined),
    cfl(-1.),
    cfl_exponent_fe_degree(1.5),
    time_step_size(-1.),
    max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
    n_refine_time(0),
    order_time_integrator(1),
    start_with_low_order(true),
    restarted_simulation(false),
    adaptive_time_stepping(false),
    restart_data(RestartData()),
    solver_info_data(SolverInfoData()),

    // SPATIAL DISCRETIZATION
    grid(GridData()),
    mapping_degree(1),
    degree_u(1),
    degree_p(1)
{
}

void
Parameters::check() const
{
  // MATHEMATICAL MODEL
  AssertThrow(formulation != Formulation::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  // PHYSICAL QUANTITIES
  AssertThrow(end_time > start_time, dealii::ExcMessage("parameter end_time must be defined"));
  AssertThrow(speed_of_sound >= 0.0, dealii::ExcMessage("parameter must be defined"));

  // TEMPORAL DISCRETIZATION
  AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
    AssertThrow(time_step_size > 0., dealii::ExcMessage("parameter must be defined"));

  if(calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    AssertThrow(cfl > 0., dealii::ExcMessage("parameter must be defined"));
    AssertThrow(cfl_exponent_fe_degree > 0., dealii::ExcMessage("cfl_exponent_fe_degree > 0."));
  }

  // SPATIAL DISCRETIZATION
  grid.check();
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
}

void
Parameters::print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Formulation", formulation);
  print_parameter(pcout, "Right-hand side", right_hand_side);
}


void
Parameters::print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Physical quantities:" << std::endl;

  // start and end time
  print_parameter(pcout, "Start time", start_time);
  print_parameter(pcout, "End time", end_time);

  // speed of sound
  print_parameter(pcout, "Speed of sound", speed_of_sound);
}

void
Parameters::print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  print_parameter(pcout, "Calculation of time step size", calculation_of_time_step_size);

  // here we do not print quantities such as time_step_size because this is
  // done by the time integration scheme (or the functions that calculate
  // the time step size)

  print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);
  print_parameter(pcout, "Temporal refinements", n_refine_time);
  print_parameter(pcout, "Order of time integration scheme", order_time_integrator);
  print_parameter(pcout, "Start with low order method", start_with_low_order);

  // restart
  print_parameter(pcout, "Restarted simulation", restarted_simulation);
  restart_data.print(pcout);

  // adaptive time-stepping
  print_parameter(pcout, "Adaptive time stepping", adaptive_time_stepping);
}

void
Parameters::print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial discretization:" << std::endl;

  grid.print(pcout);

  print_parameter(pcout, "Mapping degree", mapping_degree);

  print_parameter(pcout, "Polynomial degree pressure", degree_p);
  print_parameter(pcout, "Polynomial degree velocity", degree_u);
}

} // namespace Acoustics
} // namespace ExaDG
