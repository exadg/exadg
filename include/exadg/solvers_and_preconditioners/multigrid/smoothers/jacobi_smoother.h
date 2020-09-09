/*
 * jacobi_smoother.h
 *
 *  Created on: 2017 M03 8
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_

#include <exadg/solvers_and_preconditioners/multigrid/multigrid_input_parameters.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/smoother_base.h>
#include <exadg/solvers_and_preconditioners/preconditioner/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h>

namespace ExaDG
{
template<typename Operator, typename VectorType>
class JacobiSmoother : public SmootherBase<VectorType>
{
public:
  JacobiSmoother() : underlying_operator(nullptr), preconditioner(nullptr)
  {
  }

  ~JacobiSmoother()
  {
    delete preconditioner;
    preconditioner = nullptr;
  }

  JacobiSmoother(JacobiSmoother const &) = delete;

  JacobiSmoother &
  operator=(JacobiSmoother const &) = delete;

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
      : preconditioner(PreconditionerSmoother::PointJacobi),
        number_of_smoothing_steps(5),
        damping_factor(1.0)
    {
    }

    // preconditioner
    PreconditionerSmoother preconditioner;

    // number of iterations per smoothing step
    unsigned int number_of_smoothing_steps;

    // damping factor
    double damping_factor;
  };

  void
  initialize(Operator & operator_in, AdditionalData const & additional_data_in)
  {
    underlying_operator = &operator_in;

    data = additional_data_in;

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      preconditioner = new JacobiPreconditioner<Operator>(*underlying_operator);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      preconditioner = new BlockJacobiPreconditioner<Operator>(*underlying_operator);
    }
    else
    {
      AssertThrow(data.preconditioner == PreconditionerSmoother::PointJacobi ||
                    data.preconditioner == PreconditionerSmoother::BlockJacobi,
                  ExcMessage(
                    "Specified type of preconditioner for Jacobi smoother not implemented."));
    }
  }

  void
  update()
  {
    if(preconditioner != nullptr)
      preconditioner->update();
  }

  /*
   *  Approximately solve linear system of equations (b=src, x=dst)
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
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    VectorType tmp(src), residual(src);

    for(unsigned int k = 0; k < data.number_of_smoothing_steps; ++k)
    {
      if(k > 0)
      {
        // calculate residual r^{k} = src - A * x^{k}
        underlying_operator->vmult(residual, dst);
        residual.sadd(-1.0, 1.0, src);
      }
      else // we do not have to evaluate the residual for k=0 since dst = 0
      {
        residual = src;
      }

      // apply preconditioner: tmp = P^{-1} * residual
      preconditioner->vmult(tmp, residual);

      // x^{k+1} = x^{k} + damping_factor * tmp
      dst.add(data.damping_factor, tmp);
    }
  }

  void
  step(VectorType & dst, VectorType const & src) const
  {
    VectorType tmp(src), residual(src);

    for(unsigned int k = 0; k < data.number_of_smoothing_steps; ++k)
    {
      // calculate residual r^{k} = src - A * x^{k}
      underlying_operator->vmult(residual, dst);
      residual.sadd(-1.0, 1.0, src);

      // apply preconditioner: tmp = P^{-1} * residual
      preconditioner->vmult(tmp, residual);

      // x^{k+1} = x^{k} + damping_factor * tmp
      dst.add(data.damping_factor, tmp);
    }
  }

private:
  Operator * underlying_operator;

  AdditionalData data;

  PreconditionerBase<typename Operator::value_type> * preconditioner;
};
} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_ */
