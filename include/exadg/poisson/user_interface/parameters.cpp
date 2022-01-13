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
#include <exadg/poisson/user_interface/parameters.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

Parameters::Parameters()
  : // MATHEMATICAL MODEL
    right_hand_side(false),

    // SPATIAL DISCRETIZATION
    grid(GridData()),
    spatial_discretization(SpatialDiscretization::Undefined),
    degree(1),
    IP_factor(1.0),

    // SOLVER
    solver(Solver::Undefined),
    solver_data(SolverData(1e4, 1.e-20, 1.e-12)),
    compute_performance_metrics(false),
    preconditioner(Preconditioner::Undefined),
    multigrid_data(MultigridData()),
    enable_cell_based_face_loops(false)
{
}

void
Parameters::check() const
{
  // MATHEMATICAL MODEL

  // SPATIAL DISCRETIZATION
  grid.check();

  AssertThrow(spatial_discretization != SpatialDiscretization::Undefined,
              ExcMessage("parameter must be defined."));

  AssertThrow(degree > 0, ExcMessage("Polynomial degree must be larger than zero."));

  // SOLVER
  AssertThrow(solver != Solver::Undefined, ExcMessage("parameter must be defined."));
  AssertThrow(preconditioner != Preconditioner::Undefined,
              ExcMessage("parameter must be defined."));
}

void
Parameters::print(ConditionalOStream const & pcout, std::string const & name) const
{
  pcout << std::endl << name << std::endl;

  // MATHEMATICAL MODEL
  print_parameters_mathematical_model(pcout);

  // SPATIAL DISCRETIZATION
  print_parameters_spatial_discretization(pcout);

  // SOLVER
  print_parameters_solver(pcout);

  // NUMERICAL PARAMETERS
  print_parameters_numerical_parameters(pcout);
}

void
Parameters::print_parameters_mathematical_model(ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Right-hand side", right_hand_side);
}

void
Parameters::print_parameters_spatial_discretization(ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  grid.print(pcout);

  print_parameter(pcout, "Element type", enum_to_string(spatial_discretization));

  print_parameter(pcout, "Polynomial degree", degree);

  if(spatial_discretization == SpatialDiscretization::DG)
    print_parameter(pcout, "IP factor", IP_factor);
}

void
Parameters::print_parameters_solver(ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Solver:" << std::endl;

  print_parameter(pcout, "Solver", enum_to_string(solver));

  solver_data.print(pcout);

  print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner));

  if(preconditioner == Preconditioner::Multigrid)
    multigrid_data.print(pcout);
}


void
Parameters::print_parameters_numerical_parameters(ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Enable cell-based face loops", enable_cell_based_face_loops);
}


} // namespace Poisson
} // namespace ExaDG
