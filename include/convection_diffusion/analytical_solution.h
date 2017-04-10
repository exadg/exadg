/*
 * AnalyticalSolutionConvDiff.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_ANALYTICAL_SOLUTION_H_
#define INCLUDE_CONVECTION_DIFFUSION_ANALYTICAL_SOLUTION_H_

#include <deal.II/base/function.h>

using namespace dealii;

template<int dim>
struct AnalyticalSolutionConvDiff
{
  std::shared_ptr<Function<dim> > solution;
};


#endif /* INCLUDE_CONVECTION_DIFFUSION_ANALYTICAL_SOLUTION_H_ */
