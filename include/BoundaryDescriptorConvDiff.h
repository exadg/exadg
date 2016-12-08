/*
 * BoundaryDescriptorConvDiff.h
 *
 *  Created on: Aug 3, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_BOUNDARYDESCRIPTORCONVDIFF_H_
#define INCLUDE_BOUNDARYDESCRIPTORCONVDIFF_H_

using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

template<int dim>
struct BoundaryDescriptorConvDiff
{
  std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > > dirichlet_bc;
  std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > > neumann_bc;
};


#endif /* INCLUDE_BOUNDARYDESCRIPTORCONVDIFF_H_ */
