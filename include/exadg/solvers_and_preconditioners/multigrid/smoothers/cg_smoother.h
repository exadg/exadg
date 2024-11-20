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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_CG_SMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_CG_SMOOTHER_H_

// deal.II
#include <deal.II/lac/solver_cg.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/smoother_base.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
template<typename Operator, typename VectorType>
class CGSmoother : public SmootherBase<VectorType>
{
public:
  CGSmoother() : underlying_operator(nullptr), preconditioner(nullptr)
  {
  }

  ~CGSmoother()
  {
    delete preconditioner;
    preconditioner = nullptr;
  }

  CGSmoother(CGSmoother const &) = delete;

  CGSmoother &
  operator=(CGSmoother const &) = delete;

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData() : preconditioner(PreconditionerSmoother::None), number_of_iterations(5)
    {
    }

    // preconditioner
    PreconditionerSmoother preconditioner;

    // number of CG iterations per smoothing step
    unsigned int number_of_iterations;
  };

  void
  setup(Operator const &       operator_in,
        bool const             initialize_preconditioner,
        AdditionalData const & additional_data_in)
  {
    underlying_operator = &operator_in;
    data                = additional_data_in;

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      preconditioner =
        new JacobiPreconditioner<Operator>(*underlying_operator, initialize_preconditioner);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      preconditioner =
        new BlockJacobiPreconditioner<Operator>(*underlying_operator, initialize_preconditioner);
    }
    else
    {
      AssertThrow(data.preconditioner == PreconditionerSmoother::None,
                  dealii::ExcMessage("Specified preconditioner not implemented for CG smoother"));
    }
  }

  void
  update() final
  {
    if(preconditioner != nullptr)
      preconditioner->update();
  }

  // same as step(), but sets dst-vector to zero
  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    dst = 0.0;
    step(dst, src);
  }

  void
  step(VectorType & dst, VectorType const & src) const final
  {
    dealii::IterationNumberControl control(data.number_of_iterations, 1.e-20);

    dealii::SolverCG<VectorType> solver(control);

    if(preconditioner != nullptr)
      solver.solve(*underlying_operator, dst, src, *preconditioner);
    else
      solver.solve(*underlying_operator, dst, src, dealii::PreconditionIdentity());
  }

private:
  Operator const * underlying_operator;
  AdditionalData   data;

  PreconditionerBase<typename Operator::value_type> * preconditioner;
};
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_CG_SMOOTHER_H_ */
