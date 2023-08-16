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
enum class PreconditionerMass
{
  None,
  PointJacobi
};

struct InverseMassSolverParameters
{
  InverseMassSolverParameters()
    : implement_block_diagonal_solver_matrix_free(true),
      preconditioner(PreconditionerMass::PointJacobi),
      solver_data(SolverData(1000, 1e-12, 1e-12))
  {
  }

  // This parameter is only relevant if the mass matrix is block-diagonal, i.e. for a
  // DG formulation, and if an explicit matrix-free inverse of the mass operator is not
  // available (e.g. simplex elements). When activating this parameter, an assembly and
  // inversion of block matrices is replaced by an elementwise Krylov solver with
  // matrix-free evaluation.
  bool implement_block_diagonal_solver_matrix_free;

  // This parameter is only relevant if the mass operator is inverted by an iterative solver with
  // matrix-free implementation.
  PreconditionerMass preconditioner;

  // solver data for iterative solver
  SolverData solver_data;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_OPERATORS_INVERSE_MASS_PARAMETERS_H_ */
