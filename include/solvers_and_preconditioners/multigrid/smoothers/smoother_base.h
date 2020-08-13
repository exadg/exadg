/*
 * SmootherBase.h
 *
 *  Created on: Nov 21, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_SMOOTHER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_SMOOTHER_BASE_H_

namespace ExaDG
{
template<typename VectorType>
class SmootherBase
{
public:
  virtual ~SmootherBase()
  {
  }

  virtual void
  vmult(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  step(VectorType & dst, VectorType const & src) const = 0;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_SMOOTHER_BASE_H_ */
