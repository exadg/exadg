/*
 * verify_boundary_conditions.h
 *
 *  Created on: Oct 15, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_VERIFY_BOUNDARY_CONDITIONS_H_
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_VERIFY_BOUNDARY_CONDITIONS_H_

namespace ConvDiff
{
template<typename OperatorData>
void
do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                              OperatorData const &                 operator_data,
                              std::set<types::boundary_id> const & periodic_boundary_ids)
{
  unsigned int counter = 0;
  if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
    counter++;

  if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
    counter++;

  if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
    counter++;

  AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
}

} // namespace ConvDiff
#endif /* INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_VERIFY_BOUNDARY_CONDITIONS_H_ \
        */
