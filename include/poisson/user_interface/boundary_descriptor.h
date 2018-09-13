/*
 * boundary_descriptor.h
 *
 *  Created on:
 *      Author:
 */

#ifndef INCLUDE_LAPLACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_LAPLACE_BOUNDARY_DESCRIPTOR_H_

using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

namespace Poisson
{
template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;
};

} // namespace Poisson

#endif /* INCLUDE_CONVECTION_DIFFUSION_BOUNDARY_DESCRIPTOR_H_ */
