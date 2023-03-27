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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NONLINEAR_SOLVER_DATA_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NONLINEAR_SOLVER_DATA_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/nonlinear_solvers/enum_types.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace NonlinearSolver
{
struct SolverData
{
  SolverData()
    : max_iter(100), abs_tol(1.e-12), rel_tol(1.e-12), nonlinear_solver_type(SolverType::Newton)
  {
  }

  SolverData(unsigned int const max_iter_,
             double const       abs_tol_,
             double const       rel_tol_,
             SolverType         nonlinear_solver_type_)
    : max_iter(max_iter_),
      abs_tol(abs_tol_),
      rel_tol(rel_tol_),
      nonlinear_solver_type(nonlinear_solver_type_)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Solver type", enum_to_string(nonlinear_solver_type));
    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;
  SolverType   nonlinear_solver_type;
};

struct UpdateData
{
  UpdateData() : do_update(true), update_every_nonlinear_iter(1), update_once_converged(false)
  {
  }

  bool         do_update;
  unsigned int update_every_nonlinear_iter;
  bool         update_once_converged;
};
} // namespace NonlinearSolver
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NONLINEAR_SOLVER_DATA_H_ */
