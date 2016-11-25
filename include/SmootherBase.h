/*
 * SmootherBase.h
 *
 *  Created on: Nov 21, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SMOOTHERBASE_H_
#define INCLUDE_SMOOTHERBASE_H_

template<typename VectorType>
class SmootherBase
{
public:
  virtual ~SmootherBase(){}

  virtual void vmult(VectorType       &dst,
                     VectorType const &src) const = 0;
};


#endif /* INCLUDE_SMOOTHERBASE_H_ */
