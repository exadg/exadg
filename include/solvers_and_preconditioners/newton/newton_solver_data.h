/*
 * newton_solver_data.h
 *
 *  Created on: Nov 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_

#include "deal.II/base/conditional_ostream.h"
#include "../../functionalities/print_functions.h"

struct NewtonSolverData
{
  NewtonSolverData() : max_iter(100), abs_tol(1.e-20), rel_tol(1.e-12)
  {
  }

  NewtonSolverData(unsigned int const max_iter_, double const abs_tol_, double const rel_tol_)
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


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_ */
