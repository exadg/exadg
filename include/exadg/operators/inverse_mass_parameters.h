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

#ifndef INCLUDE_EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_
#define INCLUDE_EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_

// ExaDG
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>

namespace ExaDG
{
enum class InverseMassType
{
  MatrixfreeOperator, // currently only available via deal.II for Hypercube elements with n_nodes_1d
                      // = n_q_points_1d
  ElementwiseKrylovSolver,
  BlockMatrices
};

enum class PreconditionerMass
{
  None,
  PointJacobi
};

/**
 * Data struct for mass operator inversion in case of discontinuous Galerkin methods with a
 * block-diagonal mass matrix.
 */
struct InverseMassSolverParameters
{
  InverseMassSolverParameters()
    : implementation_type(InverseMassType::MatrixfreeOperator),
      preconditioner(PreconditionerMass::PointJacobi),
      solver_data(SolverData(1000, 1e-12, 1e-12))
  {
  }

  // The implementation type used to invert the mass operator.
  InverseMassType implementation_type;

  // This parameter is only relevant if the mass operator is inverted by an iterative solver with
  // matrix-free implementation, InverseMassType::ElementwiseKrylovSolver.
  PreconditionerMass preconditioner;

  // solver data for iterative solver in case of implementation type
  // InverseMassType::ElementwiseKrylovSolver.
  SolverData solver_data;
};

/**
 * Data struct for mass operator inversion by iterative solution techniques in case of
 * H(div)-conforming discretization where the mass matrix is a globally coupled problem as opposed
 * to DG methods (where the mass matrix is block-diagonal).
 */
struct InverseMassSolverParametersHdiv
{
  InverseMassSolverParametersHdiv()
    : preconditioner(PreconditionerMass::PointJacobi), solver_data(SolverData(1000, 1e-12, 1e-12))
  {
  }

  // The preconditioner used to iteratively solve the global mass problem
  PreconditionerMass preconditioner;

  // solver data for iterative solver
  SolverData solver_data;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_ */
