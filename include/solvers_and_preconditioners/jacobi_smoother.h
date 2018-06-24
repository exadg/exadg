/*
 * BlockJacobiSmoother.h
 *
 *  Created on: 2017 M03 8
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_


#include "block_jacobi_preconditioner.h"
#include "smoother_base.h"
#include "multigrid_input_parameters.h"
#include "jacobi_preconditioner.h"

template<int dim, typename Operator, typename VectorType>
class JacobiSmoother : public SmootherBase<VectorType>
{
public:
  JacobiSmoother()
    :
    underlying_operator(nullptr),
    preconditioner(nullptr)
  {}

  ~JacobiSmoother()
  {
    delete preconditioner;
    preconditioner = nullptr;
  }

  JacobiSmoother(JacobiSmoother const &) = delete;
  JacobiSmoother & operator=(JacobiSmoother const &) = delete;

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
     :
     preconditioner(PreconditionerJacobiSmoother::Undefined),
     number_of_smoothing_steps(5),
     damping_factor(1.0)
    {}

    // preconditioner
    PreconditionerJacobiSmoother preconditioner;

    // number of iterations per smoothing step
    unsigned int number_of_smoothing_steps;

    // damping factor
    double damping_factor;
  };

  void initialize(Operator &operator_in, AdditionalData const &additional_data_in)
  {
    underlying_operator = &operator_in;
    data = additional_data_in;

    if(data.preconditioner == PreconditionerJacobiSmoother::PointJacobi)
    {
      preconditioner = new JacobiPreconditioner<typename Operator::value_type,Operator>(*underlying_operator);
    }
    else if(data.preconditioner == PreconditionerJacobiSmoother::BlockJacobi)
    {
      preconditioner = new BlockJacobiPreconditioner<typename Operator::value_type,Operator>(*underlying_operator);
    }
    else
    {
      AssertThrow(data.preconditioner == PreconditionerJacobiSmoother::PointJacobi ||
                  data.preconditioner == PreconditionerJacobiSmoother::BlockJacobi,
          ExcMessage("Specified type of preconditioner for Jacobi smoother not implemented."));
    }
  }

  void update()
  {
    if(preconditioner != nullptr)
      preconditioner->update(underlying_operator);
  }

  /*
   *  Approximately solve linear system of equations
   *
   *    A*x = b   (r=b-A*x)
   *
   *  using the iteration
   *
   *    x^{k+1} = x^{k} + omega * P^{-1} * r^{k}
   *
   *  where
   *
   *    omega: damping factor
   *    P:     preconditioner
   */
  void vmult(VectorType       &dst,
             VectorType const &src) const
  {
    VectorType tmp(src), residual(src);

    // residual = src - A * x^{0} =  src (since initial guess x^{0} = 0)
    residual = src;

    // set dst=0 since we want to add to the dst-vector: dst += ...
    dst = 0;

    for(unsigned int k=0; k < data.number_of_smoothing_steps; ++k)
    {
      // apply preconditioner: tmp = P^{-1} * residual
      preconditioner->vmult(tmp,residual);

      // x^{k+1} = x^{k} + damping_factor * tmp
      dst.add(data.damping_factor,tmp);

      // calculate new residual r^{k+1}
      underlying_operator->vmult(residual,dst);
      residual.sadd(-1.0,1.0,src);
    }
  }

private:
  Operator *underlying_operator;
  AdditionalData data;

  PreconditionerBase<typename Operator::value_type> *preconditioner;
};



#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_ */
