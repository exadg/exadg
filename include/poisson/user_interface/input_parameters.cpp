/*
 * input_parameters.cpp
 *
 *  Created on: May 14, 2019
 *      Author: fehn
 */

#include "input_parameters.h"

namespace Poisson
{
InputParameters::InputParameters()
  : // MATHEMATICAL MODEL
    dim(2),
    right_hand_side(false),

    // SPATIAL DISCRETIZATION
    triangulation_type(TriangulationType::Undefined),
    degree(1),
    mapping(MappingType::Affine),
    h_refinements(0),
    spatial_discretization(SpatialDiscretization::Undefined),
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
InputParameters::check_input_parameters()
{
  // MATHEMATICAL MODEL
  AssertThrow(dim == 2 || dim == 3, ExcMessage("Invalid parameter."));

  // SPATIAL DISCRETIZATION
  AssertThrow(triangulation_type != TriangulationType::Undefined,
              ExcMessage("parameter must be defined."));

  AssertThrow(degree > 0, ExcMessage("Invalid parameter."));

  AssertThrow(spatial_discretization != SpatialDiscretization::Undefined,
              ExcMessage("parameter must be defined."));

  // SOLVER
  AssertThrow(solver != Solver::Undefined, ExcMessage("parameter must be defined."));
  AssertThrow(preconditioner != Preconditioner::Undefined,
              ExcMessage("parameter must be defined."));
}

void
InputParameters::print(ConditionalOStream & pcout, std::string const & name)
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
InputParameters::print_parameters_mathematical_model(ConditionalOStream & pcout)
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Space dimensions", dim);
  print_parameter(pcout, "Right-hand side", right_hand_side);
}

void
InputParameters::print_parameters_spatial_discretization(ConditionalOStream & pcout)
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));

  print_parameter(pcout, "Polynomial degree of shape functions", degree);

  print_parameter(pcout, "Mapping", enum_to_string(mapping));

  print_parameter(pcout, "Number of h-refinements", h_refinements);

  print_parameter(pcout, "Element type", enum_to_string(spatial_discretization));

  if(spatial_discretization == SpatialDiscretization::DG)
    print_parameter(pcout, "IP factor", IP_factor);
}

void
InputParameters::print_parameters_solver(ConditionalOStream & pcout)
{
  pcout << std::endl << "Solver:" << std::endl;

  print_parameter(pcout, "Solver", enum_to_string(solver));

  solver_data.print(pcout);

  print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner));

  if(preconditioner == Preconditioner::Multigrid)
    multigrid_data.print(pcout);
}


void
InputParameters::print_parameters_numerical_parameters(ConditionalOStream & pcout)
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Enable cell-based face loops", enable_cell_based_face_loops);
}


} // namespace Poisson
