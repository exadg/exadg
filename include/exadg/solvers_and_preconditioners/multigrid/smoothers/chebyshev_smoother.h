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
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/smoother_base.h>

namespace ExaDG
{
template<typename Operator, typename VectorType>
class ChebyshevSmoother : public SmootherBase<VectorType>
{
public:
  typedef
    typename dealii::PreconditionChebyshev<Operator, VectorType>::AdditionalData AdditionalData;

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
  dealii::PreconditionChebyshev<Operator, VectorType> smoother_object;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHEBYSHEVSMOOTHER_H_ */
