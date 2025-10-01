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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_DATA_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_DATA_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Newton
{
struct SolverData
{
  SolverData() : max_iter(100), abs_tol(1.e-12), rel_tol(1.e-12)
  {
  }

  SolverData(unsigned int const max_iter_, double const abs_tol_, double const rel_tol_)
    : max_iter(max_iter_), abs_tol(abs_tol_), rel_tol(rel_tol_)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;
};

struct UpdateData
{
  UpdateData() : do_update(true), update_every_newton_iter(1), update_once_converged(false)
  {
  }

  bool         do_update;
  unsigned int update_every_newton_iter;
  bool         update_once_converged;
};

} // namespace Newton
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_DATA_H_ */
