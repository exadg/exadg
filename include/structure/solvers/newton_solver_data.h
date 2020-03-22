#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_

// TODO this code seems to be copied from include/solvers_and_preconditioners/newton/

// deal.II
#include <deal.II/base/conditional_ostream.h>

struct NewtonSolverData
{
  NewtonSolverData() : max_iter(100), abs_tol(1.e-20), rel_tol(1.e-5), write_debug_vtk(false)
  {
  }

  NewtonSolverData(unsigned int const max_iter_, double const abs_tol_, double const rel_tol_)
    : max_iter(max_iter_), abs_tol(abs_tol_), rel_tol(rel_tol_), write_debug_vtk(false)
  {
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;
  bool         write_debug_vtk;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_NEWTON_SOLVER_DATA_H_ */
