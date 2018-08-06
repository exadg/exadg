/*
 * MatrixOperatorBase.h
 *
 *  Created on: Oct 28, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_MATRIXOPERATORBASE_H_
#define INCLUDE_MATRIXOPERATORBASE_H_

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/exceptions.h>

using namespace dealii;

/*
 *  Interface class needed for update of preconditioners
 */
class MatrixOperatorBase : public dealii::Subscriptor
{
public:
  MatrixOperatorBase()
  :
  dealii::Subscriptor()
  {}

  virtual ~MatrixOperatorBase(){}
  
private:
};


#endif /* INCLUDE_MATRIXOPERATORBASE_H_ */
