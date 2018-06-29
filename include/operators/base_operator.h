/*
 * BaseOperator.h
 *
 *  Created on: Oct 19, 2016
 *      Author: krank
 */

#ifndef INCLUDE_OPERATORS_BASEOPERATOR_H_
#define INCLUDE_OPERATORS_BASEOPERATOR_H_

#include "../incompressible_navier_stokes/infrastructure/fe_parameters.h"
#include "operators/matrix_operator_base.h"

namespace IncNS
{

template <int dim>
class BaseOperator: public MatrixOperatorBase
{
public:
  BaseOperator()
    :
    fe_param(nullptr)
  {}

  void set_fe_param(FEParameters<dim> * fe_param_in)
  {
    fe_param = fe_param_in;
  }
protected:
  FEParameters<dim> * fe_param;
};


}

#endif /* INCLUDE_OPERATORS_BASEOPERATOR_H_ */
