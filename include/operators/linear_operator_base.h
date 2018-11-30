/*
 * linear_operator_base.h
 *
 *  Created on: Oct 28, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_LINEAROPERATORBASE_H_
#define INCLUDE_LINEAROPERATORBASE_H_

#include <deal.II/base/subscriptor.h>

/*
 *  Interface class needed for update of preconditioners
 */
class LinearOperatorBase : public dealii::Subscriptor
{
public:
  LinearOperatorBase() : dealii::Subscriptor()
  {
  }

  virtual ~LinearOperatorBase()
  {
  }

private:
};


#endif /* INCLUDE_LINEAROPERATORBASE_H_ */
