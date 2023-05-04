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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBIPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBIPRECONDITIONER_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
template<typename Operator>
class JacobiPreconditioner : public PreconditionerBase<typename Operator::value_type>
{
public:
  typedef typename PreconditionerBase<typename Operator::value_type>::VectorType VectorType;

  JacobiPreconditioner(Operator const & underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {
    underlying_operator.initialize_dof_vector(inverse_diagonal);

    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    if(not dealii::PointerComparison::equal(&dst, &src))
      dst = src;
    dst.scale(inverse_diagonal);
  }

  void
  update() final
  {
    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }

  unsigned int
  get_size_of_diagonal()
  {
    return inverse_diagonal.size();
  }

private:
  Operator const & underlying_operator;

  VectorType inverse_diagonal;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_JACOBIPRECONDITIONER_H_ */
