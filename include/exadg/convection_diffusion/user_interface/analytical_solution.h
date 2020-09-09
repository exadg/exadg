/*
 * AnalyticalSolutionConvDiff.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_ANALYTICAL_SOLUTION_H_
#define INCLUDE_CONVECTION_DIFFUSION_ANALYTICAL_SOLUTION_H_

#include <deal.II/base/function.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim>
struct AnalyticalSolution
{
  std::shared_ptr<Function<dim>> solution;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_ANALYTICAL_SOLUTION_H_ */
