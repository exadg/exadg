/*
 * boundary_descriptor_conv_diff.h
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
enum class BoundaryType
{
  undefined,
  dirichlet,
  neumann
};

template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  // returns the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::dirichlet;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::neumann;

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryType::undefined;
  }
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_BOUNDARY_DESCRIPTOR_H_ */
