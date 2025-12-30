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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
enum class LinearSolver
{
  Undefined,
  CG,
  GMRES,
  FGMRES,
  BiCGStab,
  MinRes,
  Richardson
};

// Function to convert the `Linearsolver` enum to the `std::string` identifying the solver type in
// `dealii::SolverSelector`.
inline std::string
linear_solver_to_string(LinearSolver const linear_solver)
{
  std::string solver_name = "conversion from `LinearSolver` to `std::string` failed";

  if(linear_solver == LinearSolver::Undefined)
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Linear solver type `LinearSolver::Undefined` cannot not be parsed to "
                  "`std::string`. Select an admissible `LinearSolver`."));
  }
  else if(linear_solver == LinearSolver::CG)
  {
    solver_name = "cg";
  }
  else if(linear_solver == LinearSolver::GMRES)
  {
    solver_name = "gmres";
  }
  else if(linear_solver == LinearSolver::FGMRES)
  {
    solver_name = "fgmres";
  }
  else if(linear_solver == LinearSolver::BiCGStab)
  {
    solver_name = "bicgstab";
  }
  else if(linear_solver == LinearSolver::MinRes)
  {
    solver_name = "minres";
  }
  else if(linear_solver == LinearSolver::Richardson)
  {
    solver_name = "richardson";
  }
  else
  {
    AssertThrow(
      false, dealii::ExcMessage("This linear solver type cannot not be parsed to `std::string`."));
  }

  return solver_name;
}

struct SolverData
{
  SolverData()
    : max_iter(1e3),
      abs_tol(1e-20),
      rel_tol(1e-6),
      linear_solver(LinearSolver::Undefined),
      max_krylov_size(30)
  {
  }

  SolverData(unsigned int const max_iter_in,
             double const       abs_tol_in,
             double const       rel_tol_in,
             LinearSolver const linear_solver_in   = LinearSolver::Undefined,
             unsigned int const max_krylov_size_in = 30)
    : max_iter(max_iter_in),
      abs_tol(abs_tol_in),
      rel_tol(rel_tol_in),
      linear_solver(linear_solver_in),
      max_krylov_size(max_krylov_size_in)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    // `SolverData` can also be used to control tolerances without specifying the solver type.
    if(linear_solver != LinearSolver::Undefined)
    {
      print_parameter(pcout, "Solver", linear_solver);
    }

    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);

    // Print maximum Krylov space size for relevant methods or when using the default
    // `LinearSolver`.
    if(linear_solver == LinearSolver::FGMRES or linear_solver == LinearSolver::GMRES or
       linear_solver == LinearSolver::Undefined)
    {
      print_parameter(pcout, "Maximum size of Krylov space", max_krylov_size);
    }
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;

  // solver type to be used
  LinearSolver linear_solver;

  // only relevant for GMRES type solvers
  unsigned int max_krylov_size;
};

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_ */
