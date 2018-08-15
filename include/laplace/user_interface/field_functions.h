/*
 * field_functions.h
 *
 *  Created on: 
 *      Author: 
 */

#ifndef INCLUDE_LAPLACE_FIELD_FUNCTIONS_H_
#define INCLUDE_LAPLACE_FIELD_FUNCTIONS_H_

namespace Poisson
{
template<int dim>
struct FieldFunctions
{
  std::shared_ptr<Function<dim>> analytical_solution;
  std::shared_ptr<Function<dim>> right_hand_side;
};

} // namespace Poisson

#endif /* INCLUDE_LAPLACE_FIELD_FUNCTIONS_H_ */
