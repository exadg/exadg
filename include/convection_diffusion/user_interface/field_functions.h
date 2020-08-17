/*
 * FieldFunctionsConvDiff.h
 *
 *  Created on: Aug 3, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_
#define INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim>
struct FieldFunctions
{
  std::shared_ptr<Function<dim>> initial_solution;
  std::shared_ptr<Function<dim>> right_hand_side;
  std::shared_ptr<Function<dim>> velocity;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_ */
