/*
 * analytical_solution.h
 *
 *  Created on: 
 *      Author: 
 */

#ifndef INCLUDE_LAPLACE_ANALYTICAL_SOLUTION_H_
#define INCLUDE_LAPLACE_ANALYTICAL_SOLUTION_H_

#include <deal.II/base/function.h>

using namespace dealii;

namespace Laplace
{
template<int dim>
struct AnalyticalSolution
{
  std::shared_ptr<Function<dim>> solution;
};

} // namespace Laplace

#endif /* INCLUDE_LAPLACE_ANALYTICAL_SOLUTION_H_ */
