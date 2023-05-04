/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBISMOOTHER_H_

#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/smoother_base.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

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
      AssertThrow(data.preconditioner == PreconditionerSmoother::PointJacobi or
                    data.preconditioner == PreconditionerSmoother::BlockJacobi,
                  dealii::ExcMessage(
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
  vmult(VectorType & dst, VectorType const & src) const final
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
  step(VectorType & dst, VectorType const & src) const final
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
