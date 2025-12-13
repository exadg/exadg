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
struct SolverData
{
  SolverData()
    : max_iter(1e3), abs_tol(1e-20), rel_tol(1e-6), solver_name("undefined"), max_krylov_size(30)
  {
  }

  SolverData(unsigned int const  max_iter_in,
             double const        abs_tol_in,
             double const        rel_tol_in,
             std::string const & solver_name_in     = "undefined",
             unsigned int const  max_krylov_size_in = 30)
    : max_iter(max_iter_in),
      abs_tol(abs_tol_in),
      rel_tol(rel_tol_in),
      solver_name(solver_name_in),
      max_krylov_size(max_krylov_size_in)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    // `SolverData` can also be used to control tolerances without specifying the solver type.
    if(solver_name != "undefined")
      print_parameter(pcout, "Solver", solver_name);

    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);

    // Print maximum Krylov space size for relevant method or when using the default `solver_name`.
    if(solver_name == "fgmres" or solver_name == "gmres" or solver_name == "undefined")
      print_parameter(pcout, "Maximum size of Krylov space", max_krylov_size);
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;

  // solver type to be used
  std::string solver_name;

  // only relevant for GMRES type solvers
  unsigned int max_krylov_size;
};

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_ */
