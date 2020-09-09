/*
 * analytical_solution.h
 *
 *  Created on:
 *      Author:
 */

#ifndef INCLUDE_LAPLACE_ANALYTICAL_SOLUTION_H_
#define INCLUDE_LAPLACE_ANALYTICAL_SOLUTION_H_

#include <deal.II/base/function.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<int dim>
struct AnalyticalSolution
{
  std::shared_ptr<Function<dim>> solution;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_LAPLACE_ANALYTICAL_SOLUTION_H_ */
