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

  template<class U, class V>
    void cell(U &/*dinfo*/,
              V &/* info*/) const { }

  template<class U, class V>
    void boundary(U &/*dinfo*/,
                  V &/* info*/) const{ }

  template<class U, class V>
    void face(U &/*dinfo1*/,
              U &/*dinfo2*/,
              V &/* info1*/,
              V &/* info2*/) const{ }
  
  virtual MatrixOperatorBase* get_new(unsigned int /*deg*/) const{
      AssertThrow(false, ExcMessage("MatrixOperatorBase::get_new should be overwritten!"));
      return nullptr;
  }
  
private:
};


#endif /* INCLUDE_MATRIXOPERATORBASE_H_ */
