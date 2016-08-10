/*
 * BoundaryDescriptorConvDiff.h
 *
 *  Created on: Aug 3, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_BOUNDARYDESCRIPTORCONVDIFF_H_
#define INCLUDE_BOUNDARYDESCRIPTORCONVDIFF_H_


template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > > dirichlet_bc;
  std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > > neumann_bc;
};


#endif /* INCLUDE_BOUNDARYDESCRIPTORCONVDIFF_H_ */
