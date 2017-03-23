/*
 * FieldFunctionsConvDiff.h
 *
 *  Created on: Aug 3, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_
#define INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_


template<int dim>
struct FieldFunctionsConvDiff
{
  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  std_cxx11::shared_ptr<Function<dim> > velocity;
};


#endif /* INCLUDE_CONVECTION_DIFFUSION_FIELD_FUNCTIONS_H_ */
