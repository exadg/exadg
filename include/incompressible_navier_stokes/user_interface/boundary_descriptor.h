/*
 * BoundaryDescriptorNavierStokes.h
 *
 *  Created on: Aug 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

template<int dim>
struct BoundaryDescriptorNavierStokes
{
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet_bc;
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > neumann_bc;
  // add more types of boundary conditions
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
