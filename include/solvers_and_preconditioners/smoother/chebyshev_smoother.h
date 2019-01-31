/*
 * ChebyshevSmoother.h
 *
 *  Created on: Nov 21, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_

// deal.II
#include <deal.II/lac/precondition.h>

// parent class
#include "smoother_base.h"

using namespace dealii;

template<typename Operator, typename VectorType>
class ChebyshevSmoother : public SmootherBase<VectorType>
{
public:
  typedef typename PreconditionChebyshev<Operator, VectorType>::AdditionalData AdditionalData;

  ChebyshevSmoother()
  {
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    smoother_object.vmult(dst, src);
  }

  void
  step(VectorType & dst, VectorType const & src) const
  {
    smoother_object.step(dst, src);
  }

  void
  initialize(Operator const & matrix, AdditionalData const & additional_data)
  {
    smoother_object.initialize(matrix, additional_data);
  }

private:
  PreconditionChebyshev<Operator, VectorType> smoother_object;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_ */
