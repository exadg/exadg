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

namespace IncNS
{

/*
 *
 *   Boundary conditions:
 *
 *        example          |          velocity         |               pressure
 *   ----------------------+---------------------------+------------------------------------------------
 *        inflow, no-slip  |   Dirichlet:              |  Neumann:
 *                         | prescribe g_u             | prescribe dg_u/dt in case of dual-splitting
 *   ----------------------+---------------------------+------------------------------------------------
 *        symmetry         |   Symmetry:               |  Neumann:
 *                         | no BCs to be prescribed   | prescribe dg_u/dt=0 in case of dual-splitting
 *   ----------------------+---------------------------+------------------------------------------------
 *        outflow          |   Neumann:                |  Dirichlet:
 *                         | prescribe grad(u)*n       | prescribe g_p
 *   ----------------------+---------------------------+------------------------------------------------
 *
 *
 */

enum class BoundaryTypeU{
  Undefined,
  Dirichlet,
  Neumann,
  Symmetry
};

enum class BoundaryTypeP{
  Undefined,
  Dirichlet,
  Neumann
};

template<int dim>
struct BoundaryDescriptorU
{
  // Dirichlet: prescribe all components of the velocity
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet_bc;

  // Neumann: prescribe all components of the velocity gradient in normal direction
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > neumann_bc;

  // Symmetry: prescribe velocity normal to boundary (u*n=0) and normal velocity gradient
  //       in tangential directions (grad(u)*n - [(grad(u)*n)*n] n = 0)
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > symmetry_bc;

  // add more types of boundary conditions
};

template<int dim>
struct BoundaryDescriptorP
{
  // Dirichlet: prescribe pressure value
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet_bc;

  // Neumann: do nothing in case of coupled solution approach (one only has to
  // discretize the pressure gradient for this solution approach),
  // prescribe Neumann BCs in case of projection type solution methods
  // where a Poisson equation is solved for the pressure.
  // Hence, the Function<dim> is irrelevant for the coupled solver and
  // also for the pressure-correction scheme (always homogeneous NBC also for higher-order
  // formulations).
  // BUT: Use Function<dim> to store boundary condition du/dt in case of
  //      dual splitting scheme.
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > neumann_bc;
};


}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
