/*
 * BoundaryDescriptorLaplace.h
 *
 *  Created on: Nov 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_BOUNDARY_DESCRIPTOR_LAPLACE_H_
#define INCLUDE_POISSON_BOUNDARY_DESCRIPTOR_LAPLACE_H_

using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

template<int dim>
struct BoundaryDescriptorLaplace
{
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet;
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > neumann;
};


#endif /* INCLUDE_POISSON_BOUNDARY_DESCRIPTOR_LAPLACE_H_ */
