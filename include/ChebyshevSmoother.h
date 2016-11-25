/*
 * ChebyshevSmoother.h
 *
 *  Created on: Nov 21, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CHEBYSHEVSMOOTHER_H_
#define INCLUDE_CHEBYSHEVSMOOTHER_H_


#include "SmootherBase.h"

template<typename Operator, typename VectorType>
class ChebyshevSmoother : public SmootherBase<VectorType>
{
public:
  typedef typename PreconditionChebyshev<Operator, VectorType>::AdditionalData AdditionalData;

  ChebyshevSmoother(){}

  void vmult(VectorType       &dst,
             VectorType const &src) const
  {
    smoother_object.vmult(dst,src);
  }

  void initialize(Operator const &matrix, AdditionalData const &additional_data)
  {
    smoother_object.initialize(matrix,additional_data);
  }

private:
  PreconditionChebyshev<Operator, VectorType> smoother_object;
};

#endif /* INCLUDE_CHEBYSHEVSMOOTHER_H_ */
