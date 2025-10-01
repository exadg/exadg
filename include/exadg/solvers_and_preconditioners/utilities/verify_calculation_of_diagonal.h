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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_VERIFY_CALCULATION_OF_DIAGONAL_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_VERIFY_CALCULATION_OF_DIAGONAL_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
/*
 *  To check the correctness of the efficient computation of the diagonal
 *  the result is compared to a naive calculation that simply applies the
 *  whole matrix-vector product N_dofs times. Accordingly, to call this
 *  function the Operator passed to this function has to implement a
 *  function called vmult() that calculates the matrix-vector product.
 */
template<typename Operator, typename value_type>
void
verify_calculation_of_diagonal(Operator &                                               op,
                               dealii::LinearAlgebra::distributed::Vector<value_type> & diagonal,
                               MPI_Comm const &                                         mpi_comm)
{
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);
  pcout << "Verify calculation of diagonal:" << std::endl;

  typedef dealii::LinearAlgebra::distributed::Vector<value_type> VectorType;

  VectorType diagonal_check(diagonal);
  VectorType src(diagonal);
  VectorType dst(diagonal);

  diagonal_check = 0.0;
  src            = 0.0;
  dst            = 0.0;

  /*
   *  Set dof-value i to 1.0, calculate matrix-vector
   *  product and store row i of the result in diagonal_check.
   */
  for(unsigned int i = 0; i < diagonal_check.size(); ++i)
  {
    if(diagonal_check.in_local_range(i))
    {
      src(i) = 1.0;
    }

    op.vmult(dst, src);

    if(diagonal_check.in_local_range(i))
    {
      diagonal_check(i) = dst(i);
      src(i)            = 0.0;
    }
  }

  value_type norm_diagonal       = diagonal.l2_norm();
  value_type norm_diagonal_check = diagonal_check.l2_norm();

  pcout << std::endl
        << "L2 norm diagonal - Variant 1: " << std::setprecision(10) << norm_diagonal << std::endl;
  pcout << "L2 norm diagonal - Variant 2: " << std::setprecision(10) << norm_diagonal_check
        << std::endl;

  diagonal_check.add(-1.0, diagonal);
  value_type norm_error = diagonal_check.l2_norm();

  pcout << "L2 error diagonal: " << std::setprecision(10) << norm_error << std::endl << std::endl;
}

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_VERIFY_CALCULATION_OF_DIAGONAL_H_ */
