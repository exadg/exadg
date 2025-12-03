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
  SolverData() : max_iter(1e3), abs_tol(1e-20), rel_tol(1e-6), max_krylov_size(30)
  {
  }

  SolverData(unsigned int const max_iter_in,
             double const       abs_tol_in,
             double const       rel_tol_in,
             unsigned int const max_krylov_size_in = 30)
    : max_iter(max_iter_in),
      abs_tol(abs_tol_in),
      rel_tol(rel_tol_in),
      max_krylov_size(max_krylov_size_in)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);
    print_parameter(pcout, "Maximum size of Krylov space", max_krylov_size);
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;
  // only relevant for GMRES type solvers
  unsigned int max_krylov_size;
};

// `SolverData` structs for deal.II wrapper classes
namespace Krylov
{
struct SolverDataCG
{
  SolverDataCG()
    : max_iter(1e4),
      solver_tolerance_abs(1.e-20),
      solver_tolerance_rel(1.e-6),
      use_preconditioner(false),
      compute_performance_metrics(false)
  {
  }

  unsigned int max_iter;
  double       solver_tolerance_abs;
  double       solver_tolerance_rel;
  bool         use_preconditioner;
  bool         compute_performance_metrics;
};

struct SolverDataGMRES
{
  SolverDataGMRES()
    : max_iter(1e4),
      solver_tolerance_abs(1.e-20),
      solver_tolerance_rel(1.e-6),
      use_preconditioner(false),
      max_n_tmp_vectors(30),
      compute_eigenvalues(false),
      compute_performance_metrics(false)
  {
  }

  unsigned int max_iter;
  double       solver_tolerance_abs;
  double       solver_tolerance_rel;
  bool         use_preconditioner;
  unsigned int max_n_tmp_vectors;
  bool         compute_eigenvalues;
  bool         compute_performance_metrics;
};

struct SolverDataFGMRES
{
  SolverDataFGMRES()
    : max_iter(1e4),
      solver_tolerance_abs(1.e-20),
      solver_tolerance_rel(1.e-6),
      use_preconditioner(false),
      max_n_tmp_vectors(30),
      compute_performance_metrics(false)
  {
  }

  unsigned int max_iter;
  double       solver_tolerance_abs;
  double       solver_tolerance_rel;
  bool         use_preconditioner;
  unsigned int max_n_tmp_vectors;
  bool         compute_performance_metrics;
};

} // namespace Krylov
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_ */
