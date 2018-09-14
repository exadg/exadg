/*
 * BoundaryDescriptorConvDiff.h
 *
 *  Created on: Aug 3, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_BOUNDARY_DESCRIPTOR_H_

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

using namespace dealii;

namespace ConvDiff
{
template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_BOUNDARY_DESCRIPTOR_H_ */
