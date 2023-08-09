/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_ADDITIVESCHWARZPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_ADDITIVESCHWARZPRECONDITIONER_H_

#include <deal.II/lac/sparse_matrix.h>

#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
template<typename Operator>
class AdditiveSchwarzPreconditioner : public PreconditionerBase<typename Operator::value_type>
{
public:
  typedef typename PreconditionerBase<typename Operator::value_type>::VectorType VectorType;

  AdditiveSchwarzPreconditioner(Operator const & underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {
    underlying_operator.compute_factorized_as_matrices();
  }

  /*
   *  This function updates the additive Schwarz preconditioner.
   *  Make sure that the underlying operator has been updated
   *  when calling this function.
   */
  void
  update() final
  {
    underlying_operator.compute_factorized_as_matrices();
  }

  /*
   *  This function applies additive Schwarz preconditioner.
   *  Make sure that the additive Schwarz preconditioner has been
   *  updated when calling this function.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    underlying_operator.apply_inverse_as_matrices(dst, src);
  }

private:
  Operator const & underlying_operator;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ADDITIVESCHWARZPRECONDITIONER_H_ */
