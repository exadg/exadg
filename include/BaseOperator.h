/*
 * BaseOperator.h
 *
 *  Created on: Oct 19, 2016
 *      Author: krank
 */

#ifndef INCLUDE_BASEOPERATOR_H_
#define INCLUDE_BASEOPERATOR_H_

#include "FE_Parameters.h"
#include "MatrixOperatorBase.h"

//template <int dim>
//class BaseOperator: public Subscriptor

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


#endif /* INCLUDE_BASEOPERATOR_H_ */
