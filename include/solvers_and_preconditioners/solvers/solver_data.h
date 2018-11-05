/*
 * solver_data.h
 *
 *  Created on: Oct 23, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_

#include "deal.II/base/conditional_ostream.h"
#include "../../functionalities/print_functions.h"

struct SolverData
{
  SolverData() : max_iter(1e3), abs_tol(1e-20), rel_tol(1e-6)
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


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SOLVERS_SOLVER_DATA_H_ */
