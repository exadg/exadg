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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_

#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
class InverseMassPreconditioner : public PreconditionerBase<Number>
{
public:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

  InverseMassPreconditioner(dealii::MatrixFree<dim, Number> const & matrix_free,
                            InverseMassOperatorData const           inverse_mass_operator_data)
  {
    inverse_mass_operator.initialize(matrix_free, inverse_mass_operator_data);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    inverse_mass_operator.apply(dst, src);
  }

  void
  update()
  {
    // do nothing
  }

private:
  InverseMassOperator<dim, n_components, Number> inverse_mass_operator;
};
} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_ */
