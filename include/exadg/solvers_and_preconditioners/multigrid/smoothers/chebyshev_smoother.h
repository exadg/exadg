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
#include <exadg/solvers_and_preconditioners/preconditioners/additive_schwarz_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
template<typename Operator, typename VectorType>
class ChebyshevSmoother : public SmootherBase<VectorType>
{
public:
  typedef dealii::PreconditionChebyshev<Operator, VectorType, dealii::DiagonalMatrix<VectorType>>
    ChebyshevPointJacobi_NEW;
  typedef dealii::PreconditionChebyshev<Operator, VectorType, JacobiPreconditioner<Operator>>
    ChebyshevPointJacobi_OLD;
  typedef dealii::PreconditionChebyshev<Operator, VectorType, BlockJacobiPreconditioner<Operator>>
    ChebyshevBlockJacobi;
  typedef dealii::
    PreconditionChebyshev<Operator, VectorType, AdditiveSchwarzPreconditioner<Operator>>
      ChebyshevAdditiveSchwarz;

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
      // ##+
      VectorType dst_copy = dst;
      VectorType src_copy = src;

      chebyshev_point_jacobi_NEW->vmult(dst_copy, src_copy);
      VectorType tmp = dst_copy;
      chebyshev_point_jacobi_OLD->vmult(dst, src);
      tmp -= dst;
      std::cout << "tmp.l2_norm() = " << tmp.l2_norm() << "\n";

      // use new version:
      dst = dst_copy;
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      chebyshev_block_jacobi->vmult(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      chebyshev_additive_schwarz->vmult(dst, src);
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
      // ##+
      VectorType dst_copy = dst;
      VectorType src_copy = src;

      chebyshev_point_jacobi_NEW->step(dst_copy, src_copy);
      VectorType tmp = dst_copy;
      chebyshev_point_jacobi_OLD->step(dst, src);
      tmp -= dst;
      std::cout << "tmp.l2_norm() 2 = " << tmp.l2_norm() << "\n";

      // use new version:
      dst = dst_copy;
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      chebyshev_block_jacobi->step(dst, src);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      chebyshev_additive_schwarz->step(dst, src);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  update() final
  {
    AssertThrow(underlying_operator != nullptr,
                dealii::ExcMessage("Pointer underlying_operator is uninitialized."));

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      std::cout << "UPDATE \n";
      preconditioner_point_jacobi_OLD->update();

      underlying_operator->calculate_inverse_diagonal(
        preconditioner_point_jacobi_NEW->get_vector());

      // Check if the contents are the same.
      VectorType const & inv_diag = preconditioner_point_jacobi_NEW->get_vector();

      std::cout << "inv_diag.l2_norm() = " << inv_diag.l2_norm() << " ##+ \n";

      VectorType inv_diag2 = preconditioner_point_jacobi_OLD->get_inverse_diagonal();

      inv_diag2 -= inv_diag;
      std::cout << "||inv_diag - inv_diag2||2 = " << inv_diag2.l2_norm() << " ##+ \n";

      chebyshev_point_jacobi_NEW->initialize(*underlying_operator, additional_data_point_NEW);
      chebyshev_point_jacobi_OLD->initialize(*underlying_operator, additional_data_point_OLD);
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      preconditioner_block_jacobi->update();
      chebyshev_block_jacobi->initialize(*underlying_operator, additional_data_block);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      preconditioner_additive_schwarz->update();
      chebyshev_additive_schwarz->initialize(*underlying_operator,
                                             additional_data_additive_schwarz);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  setup(Operator const &       operator_in,
        bool const             initialize_preconditioner,
        AdditionalData const & additional_data)
  {
    underlying_operator = &operator_in;
    data                = additional_data;

    if(data.preconditioner == PreconditionerSmoother::PointJacobi)
    {
      std::cout << "INITIALIZE \n";
      preconditioner_point_jacobi_NEW = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
      if(true /*initialize_preconditioner*/)
      {
        std::cout << "INITIALIZE 2 \n";
        underlying_operator->calculate_inverse_diagonal(
          preconditioner_point_jacobi_NEW->get_vector());
      }

      preconditioner_point_jacobi_OLD =
        std::make_shared<JacobiPreconditioner<Operator>>(*underlying_operator,
                                                         true /* initialize_preconditioner*/);



      // Check if the contents are the same.
      VectorType const & inv_diag = preconditioner_point_jacobi_NEW->get_vector();

      std::cout << "inv_diag.l2_norm() = " << inv_diag.l2_norm() << " ##+ \n";

      VectorType inv_diag2 = preconditioner_point_jacobi_OLD->get_inverse_diagonal();

      inv_diag2 -= inv_diag;
      std::cout << "||inv_diag - inv_diag2||2 = " << inv_diag2.l2_norm() << " ##+ \n";



      additional_data_point_NEW.preconditioner      = preconditioner_point_jacobi_NEW;
      additional_data_point_NEW.smoothing_range     = data.smoothing_range;
      additional_data_point_NEW.degree              = data.degree;
      additional_data_point_NEW.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      additional_data_point_OLD.preconditioner      = preconditioner_point_jacobi_OLD;
      additional_data_point_OLD.smoothing_range     = data.smoothing_range;
      additional_data_point_OLD.degree              = data.degree;
      additional_data_point_OLD.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      chebyshev_point_jacobi_NEW = std::make_shared<ChebyshevPointJacobi_NEW>();
      chebyshev_point_jacobi_OLD = std::make_shared<ChebyshevPointJacobi_OLD>();

      if(initialize_preconditioner)
      {
        chebyshev_point_jacobi_OLD->initialize(*underlying_operator, additional_data_point_OLD);
        chebyshev_point_jacobi_NEW->initialize(*underlying_operator, additional_data_point_NEW);
      }
    }
    else if(data.preconditioner == PreconditionerSmoother::BlockJacobi)
    {
      preconditioner_block_jacobi =
        std::make_shared<BlockJacobiPreconditioner<Operator>>(*underlying_operator,
                                                              initialize_preconditioner);

      additional_data_block.preconditioner      = preconditioner_block_jacobi;
      additional_data_block.smoothing_range     = data.smoothing_range;
      additional_data_block.degree              = data.degree;
      additional_data_block.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      chebyshev_block_jacobi = std::make_shared<ChebyshevBlockJacobi>();

      if(initialize_preconditioner)
        chebyshev_block_jacobi->initialize(*underlying_operator, additional_data_block);
    }
    else if(data.preconditioner == PreconditionerSmoother::AdditiveSchwarz)
    {
      preconditioner_additive_schwarz =
        std::make_shared<AdditiveSchwarzPreconditioner<Operator>>(*underlying_operator,
                                                                  initialize_preconditioner);

      additional_data_additive_schwarz.preconditioner      = preconditioner_additive_schwarz;
      additional_data_additive_schwarz.smoothing_range     = data.smoothing_range;
      additional_data_additive_schwarz.degree              = data.degree;
      additional_data_additive_schwarz.eig_cg_n_iterations = data.iterations_eigenvalue_estimation;

      chebyshev_additive_schwarz = std::make_shared<ChebyshevAdditiveSchwarz>();

      if(initialize_preconditioner)
        chebyshev_additive_schwarz->initialize(*underlying_operator,
                                               additional_data_additive_schwarz);
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

private:
  Operator const * underlying_operator;
  AdditionalData   data;

  std::shared_ptr<ChebyshevPointJacobi_NEW> chebyshev_point_jacobi_NEW;
  std::shared_ptr<ChebyshevPointJacobi_OLD> chebyshev_point_jacobi_OLD;
  std::shared_ptr<ChebyshevBlockJacobi>     chebyshev_block_jacobi;
  std::shared_ptr<ChebyshevAdditiveSchwarz> chebyshev_additive_schwarz;

  std::shared_ptr<dealii::DiagonalMatrix<VectorType>>      preconditioner_point_jacobi_NEW;
  std::shared_ptr<JacobiPreconditioner<Operator>>          preconditioner_point_jacobi_OLD;
  std::shared_ptr<BlockJacobiPreconditioner<Operator>>     preconditioner_block_jacobi;
  std::shared_ptr<AdditiveSchwarzPreconditioner<Operator>> preconditioner_additive_schwarz;

  typename ChebyshevPointJacobi_NEW::AdditionalData additional_data_point_NEW;
  typename ChebyshevPointJacobi_OLD::AdditionalData additional_data_point_OLD;
  typename ChebyshevBlockJacobi::AdditionalData     additional_data_block;
  typename ChebyshevAdditiveSchwarz::AdditionalData additional_data_additive_schwarz;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_ */
