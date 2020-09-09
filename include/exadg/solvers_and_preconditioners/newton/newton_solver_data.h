/*
 * newton_solver_data.h
 *
 *  Created on: Nov 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Newton
{
using namespace dealii;

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
  print(ConditionalOStream & pcout)
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
  UpdateData()
    : do_update(true),
      threshold_newton_iter(1),
      threshold_linear_iter(std::numeric_limits<unsigned int>::max())
  {
  }

  bool         do_update;
  unsigned int threshold_newton_iter;
  unsigned int threshold_linear_iter;
};
} // namespace Newton
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_ */
