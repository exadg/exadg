/*
 * MultigridInputParameters.h
 *
 *  Created on: Jun 20, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_MULTIGRIDINPUTPARAMETERS_H_
#define INCLUDE_MULTIGRIDINPUTPARAMETERS_H_


// enum class MultigridSmoother { Chebyshev };

/*
 *  Multigrid coarse grid solver
 */
enum class MultigridCoarseGridSolver
{
  coarse_chebyshev_smoother,
  coarse_iterative_nopreconditioner,
  coarse_iterative_jacobi
};

struct MultigridData
{
  MultigridData()
    :
    // multigrid_smoother(MultigridSmoother::Chebyshev),
    smoother_poly_degree(5),
    smoother_smoothing_range(20),
    coarse_solver(MultigridCoarseGridSolver::coarse_chebyshev_smoother)
  {}

  // Sets the multigrid smoother: currently only Chebyshev implemented, so there is no need for this variable
  // MultigridSmoother multigrid_smoother;

  // Sets the polynomial degree of the Chebyshev smoother (Chebyshev accelerated Jacobi smoother)
  double smoother_poly_degree;

  // Sets the smoothing range of the Chebyshev smoother
  double smoother_smoothing_range;

  // Sets the coarse grid solver
  MultigridCoarseGridSolver coarse_solver;
};


#endif /* INCLUDE_MULTIGRIDINPUTPARAMETERS_H_ */
