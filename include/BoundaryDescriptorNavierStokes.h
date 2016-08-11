/*
 * BoundaryDescriptorNavierStokes.h
 *
 *  Created on: Aug 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_BOUNDARYDESCRIPTORNAVIERSTOKES_H_
#define INCLUDE_BOUNDARYDESCRIPTORNAVIERSTOKES_H_


template<int dim>
struct BoundaryDescriptorNavierStokes
{
  std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > > dirichlet_bc;
  std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > > neumann_bc;
  // add more types of boundary conditions
};



#endif /* INCLUDE_BOUNDARYDESCRIPTORNAVIERSTOKES_H_ */
