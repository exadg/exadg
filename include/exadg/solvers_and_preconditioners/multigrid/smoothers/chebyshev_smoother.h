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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_

// deal.II
#include <deal.II/lac/precondition.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/smoother_base.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>

namespace ExaDG
{
template<typename Operator, typename VectorType>
class ChebyshevSmoother : public SmootherBase<VectorType>
{
public:
  typedef dealii::PreconditionChebyshev<Operator, VectorType, dealii::DiagonalMatrix<VectorType>>
    ChebyshevPointJacobi;
  typedef dealii::PreconditionChebyshev<Operator, VectorType, BlockJacobiPreconditioner<Operator>>
    ChebyshevBlockJacobi;

  ChebyshevSmoother() : underlying_operator(nullptr)
  {
  }

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
      : preconditioner(PreconditionerSmoother::PointJacobi),
        smoothing_range(20),
        degree(5),
        iterations_eigenvalue_estimation(20)
    {
    }

    // preconditioner
    PreconditionerSmoother preconditioner;

    // sets the smoothing range (range of eigenvalues to be smoothed)
    double smoothing_range;

    // degree of Chebyshev smoother
    unsigned int degree;

    // number of CG iterations for estimation of eigenvalues
    unsigned int iterations_eigenvalue_estimation;
  };

  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      smoother_point_jacobi->vmult(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      smoother_block_jacobi->vmult(dst, src);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  step(VectorType & dst, VectorType const & src) const final
  {
    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      smoother_point_jacobi->step(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      smoother_block_jacobi->step(dst, src);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  update()
  {
    AssertThrow(underlying_operator != nullptr,
                dealii::ExcMessage("Pointer underlying_operator is uninitialized."));

    initialize(*underlying_operator, data);
  }

  void
  initialize(Operator const & operator_in, AdditionalData const & additional_data)
  {
    underlying_operator = &operator_in;
    data                = additional_data;

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      typename ChebyshevPointJacobi::AdditionalData additional_data_dealii;

      std::shared_ptr<dealii::DiagonalMatrix<VectorType>> jacobi_preconditioner =
        std::make_shared<dealii::DiagonalMatrix<VectorType>>();
      VectorType & diagonal_vector = jacobi_preconditioner->get_vector();

      underlying_operator->initialize_dof_vector(diagonal_vector);
      underlying_operator->calculate_inverse_diagonal(diagonal_vector);

      additional_data_dealii.preconditioner      = jacobi_preconditioner;
      additional_data_dealii.smoothing_range     = data.smoothing_range;
      additional_data_dealii.degree              = data.degree;
      additional_data_dealii.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      smoother_point_jacobi = std::make_shared<ChebyshevPointJacobi>();
      smoother_point_jacobi->initialize(*underlying_operator, additional_data_dealii);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      typename ChebyshevBlockJacobi::AdditionalData additional_data_dealii;

      additional_data_dealii.preconditioner =
        std::make_shared<BlockJacobiPreconditioner<Operator>>(*underlying_operator);
      additional_data_dealii.smoothing_range     = data.smoothing_range;
      additional_data_dealii.degree              = data.degree;
      additional_data_dealii.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      smoother_block_jacobi = std::make_shared<ChebyshevBlockJacobi>();
      smoother_block_jacobi->initialize(*underlying_operator, additional_data_dealii);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

private:
  Operator const * underlying_operator;
  AdditionalData   data;

  std::shared_ptr<ChebyshevPointJacobi> smoother_point_jacobi;
  std::shared_ptr<ChebyshevBlockJacobi> smoother_block_jacobi;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_ */
