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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCK_JACOBI_MATRICES_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCK_JACOBI_MATRICES_H_

namespace ExaDG
{
/*
 *  Initialize block Jacobi matrices with zeros.
 */
template<typename Number>
void
initialize_block_jacobi_matrices_with_zero(std::vector<dealii::LAPACKFullMatrix<Number>> & matrices)
{
  // initialize matrices
  for(auto & m : matrices)
    m = 0;
}


/*
 *  This function calculates the LU factorization for a given vector
 *  of matrices of type LAPACKFullMatrix.
 */
template<typename Number>
void
calculate_lu_factorization_block_jacobi(std::vector<dealii::LAPACKFullMatrix<Number>> & matrices)
{
  for(auto & matrix : matrices)
  {
    try // the matrix might be singular
    {
      matrix.compute_lu_factorization();
    }
    catch(std::exception & exc)
    {
      // add a small, positive value to the diagonal
      // of the LU factorized matrix
      for(unsigned int i = 0; i < matrix.m(); ++i)
      {
        for(unsigned int j = 0; j < matrix.n(); ++j)
        {
          if(i == j)
            matrix(i, j) += 1.e-4;
        }
      }
    }
  }
}

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCK_JACOBI_MATRICES_H_ */
