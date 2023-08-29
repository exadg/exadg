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
#include <exadg/structure/user_interface/parameters.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Structure
{
Parameters::Parameters()
  : // MATHEMATICAL MODEL
    problem_type(ProblemType::Undefined),
    body_force(false),
    large_deformation(false),
    pull_back_body_force(false),
    pull_back_traction(false),
	spatial_integration(false),

    // PHYSICAL QUANTITIES
    density(1.0),

    // WEAK LINEAR DAMPING
    weak_damping_active(false),
    weak_damping_coefficient(0.0),

    // TEMPORAL DISCRETIZATION
    start_time(0.0),
    end_time(1.0),
    time_step_size(1.0),
    max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
    n_refine_time(0),
    gen_alpha_type(GenAlphaType::GenAlpha),
    spectral_radius(1.0),
    solver_info_data(SolverInfoData()),
    restarted_simulation(false),
    restart_data(RestartData()),

    // quasi-static solver
    load_increment(1.0),

    // SPATIAL DISCRETIZATION
    grid(GridData()),
    mapping_degree(1),
    degree(1),

    // SOLVER
    newton_solver_data(Newton::SolverData(1e4, 1.e-12, 1.e-6)),
    solver(Solver::Undefined),
    solver_data(SolverData(1e4, 1.e-12, 1.e-6, 100)),
    preconditioner(Preconditioner::AMG),
    update_preconditioner(false),
    update_preconditioner_every_time_steps(1),
    update_preconditioner_every_newton_iterations(10),
    update_preconditioner_once_newton_converged(false),
    multigrid_data(MultigridData())
{
}

void
Parameters::check() const
{
  // MATHEMATICAL MODEL
  AssertThrow(problem_type != ProblemType::Undefined,
              dealii::ExcMessage("Parameter must be defined."));

  if(problem_type == ProblemType::QuasiStatic)
  {
    AssertThrow(large_deformation == true,
                dealii::ExcMessage(
                  "QuasiStatic solver only implemented for nonlinear formulation."));
  }

  if(problem_type == ProblemType::Unsteady)
  {
    AssertThrow(restarted_simulation == false,
                dealii::ExcMessage("Restart has not been implemented."));

    if(weak_damping_active)
    {
      AssertThrow(weak_damping_coefficient > 0.0,
                  dealii::ExcMessage("Weak linear damping requires positive coefficient."));
    }
  }

  if(spatial_integration)
  {
    AssertThrow(large_deformation, dealii::ExcMessage("Spatial integration only different from material configuration for finite strain problems."));
  }

  // SPATIAL DISCRETIZATION
  grid.check();

  AssertThrow(degree > 0, dealii::ExcMessage("Polynomial degree must be larger than zero."));

  // SOLVER
  AssertThrow(solver != Solver::Undefined, dealii::ExcMessage("Parameter must be defined."));
}

bool
Parameters::involves_h_multigrid() const
{
  if(preconditioner == Preconditioner::Multigrid and multigrid_data.involves_h_transfer())
    return true;
  else
    return false;
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
  print_parameters_solver(pcout);
}

void
Parameters::print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Problem type", problem_type);

  print_parameter(pcout, "Body force", body_force);

  print_parameter(pcout, "Large deformation", large_deformation);

  if(large_deformation)
  {
    print_parameter(pcout, "Pull back body force", pull_back_body_force);
    print_parameter(pcout, "Pull back traction", pull_back_traction);
    print_parameter(pcout, "Integrate over spatial configuration", spatial_integration);
  }
}

void
Parameters::print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Physical quantities:" << std::endl;

  if(problem_type == ProblemType::Unsteady)
  {
    print_parameter(pcout, "Density", density);

    if(weak_damping_active)
    {
      print_parameter(pcout, "Weak damping coefficient", weak_damping_coefficient);
    }
  }
}

void
Parameters::print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  if(problem_type == ProblemType::QuasiStatic)
  {
    print_parameter(pcout, "load_increment", load_increment);
  }

  if(problem_type == ProblemType::Unsteady)
  {
    print_parameter(pcout, "Start time", start_time);
    print_parameter(pcout, "End time", end_time);
    print_parameter(pcout, "Max. number of time steps", max_number_of_time_steps);
    print_parameter(pcout, "Temporal refinements", n_refine_time);
    print_parameter(pcout, "Time integration type", gen_alpha_type);
    print_parameter(pcout, "Spectral radius", spectral_radius);
    solver_info_data.print(pcout);
    if(restarted_simulation)
      restart_data.print(pcout);
  }
}

void
Parameters::print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  grid.print(pcout);

  print_parameter(pcout, "Mapping degree", mapping_degree);

  print_parameter(pcout, "Polynomial degree", degree);
}

void
Parameters::print_parameters_solver(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Solver:" << std::endl;

  // nonlinear solver
  if(large_deformation)
  {
    pcout << std::endl << "Newton:" << std::endl;
    newton_solver_data.print(pcout);
  }

  // linear solver
  pcout << std::endl << "Linear solver:" << std::endl;
  print_parameter(pcout, "Solver", solver);
  solver_data.print(pcout);

  // preconditioner for linear system of equations
  print_parameter(pcout, "Preconditioner", preconditioner);

  if(preconditioner == Preconditioner::Multigrid)
  {
    multigrid_data.print(pcout);
  }
}

} // namespace Structure
} // namespace ExaDG
