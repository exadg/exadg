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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_
#define EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>

namespace ExaDG
{
enum class InverseMassType
{
  MatrixfreeOperator, // currently only available via deal.II for Hypercube elements with n_nodes_1d
                      // = n_q_points_1d
  ElementwiseKrylovSolver,
  BlockMatrices,
  GlobalKrylovSolver
};

enum class PreconditionerMass
{
  None,
  PointJacobi,
  BlockJacobi,
  AMG
};

/**
 * Data struct for mass operator inversion covering L2-conforming or Hdiv-conforming discontinuous
 * Galerkin methods and continuous Galerkin methods. Depending on the underlying discretization,
 * various implementations exploiting the structure of the matrix related to the discretized mass
 * operator are available. Choices not approximating the inverse operator up to linear solver
 * tolerance are asserted.
 */
struct InverseMassParameters
{
  InverseMassParameters()
    : implementation_type(InverseMassType::MatrixfreeOperator),
      preconditioner(PreconditionerMass::PointJacobi),
      solver_data(SolverData()),
      amg_data(AMGData())
  {
  }

  // The implementation type used to invert the mass operator.
  InverseMassType implementation_type;

  // This parameter is only relevant if the mass operator is inverted by an iterative solver with
  // matrix-free implementation, `InverseMassType::ElementwiseKrylovSolver` or
  // `InverseMassType::GlobalKrylovSolver`.
  PreconditionerMass preconditioner;

  // solver data for iterative solver in case of implementation type
  // `InverseMassType::ElementwiseKrylovSolver` or `InverseMassType::GlobalKrylovSolver`.
  SolverData solver_data;

  // Configuration of AMG settings for `PreconditionerMass::AMG`.
  AMGData amg_data;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_ */
