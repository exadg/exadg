/*
 * MultigridInputParameters.h
 *
 *  Created on: Jun 20, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_MULTIGRIDINPUTPARAMETERS_H_
#define INCLUDE_MULTIGRIDINPUTPARAMETERS_H_


enum class MultigridSmoother { Chebyshev };
enum class MultigridCoarseGridSolver { coarse_chebyshev_smoother, coarse_iterative_nopreconditioner, coarse_iterative_jacobi };


#endif /* INCLUDE_MULTIGRIDINPUTPARAMETERS_H_ */
