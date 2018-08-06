/*
 * FieldFunctionsConvDiff.h
 *
 *  Created on: Aug 3, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_
#define INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_

namespace Laplace
{
template<int dim>
struct FieldFunctions
{
  std::shared_ptr<Function<dim>> analytical_solution;
  std::shared_ptr<Function<dim>> right_hand_side;
};

} // namespace Laplace

#endif /* INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_ */
