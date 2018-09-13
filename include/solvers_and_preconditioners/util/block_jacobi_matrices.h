/*
 * block_jacobi_matrices.h
 *
 *  Created on: Mar 23, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCK_JACOBI_MATRICES_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCK_JACOBI_MATRICES_H_


/*
 *  Initialize block Jacobi matrices with zeros.
 */
template<typename Number>
void
initialize_block_jacobi_matrices_with_zero(std::vector<LAPACKFullMatrix<Number>> & matrices)
{
  // initialize matrices
  for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin();
      it != matrices.end();
      ++it)
  {
    *it = 0;
  }
}


/*
 *  This function calculates the LU factorization for a given vector
 *  of matrices of type LAPACKFullMatrix.
 */
template<typename Number>
void
calculate_lu_factorization_block_jacobi(std::vector<LAPACKFullMatrix<Number>> & matrices)
{
  for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin();
      it != matrices.end();
      ++it)
  {
    LAPACKFullMatrix<Number> copy(*it);
    try // the matrix might be singular
    {
      (*it).compute_lu_factorization();
    }
    catch(std::exception & exc)
    {
      // add a small, positive value to the diagonal
      // of the LU factorized matrix
      for(unsigned int i = 0; i < (*it).m(); ++i)
      {
        for(unsigned int j = 0; j < (*it).n(); ++j)
        {
          if(i == j)
            (*it)(i, j) += 1.e-4;
        }
      }
    }
  }
}


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_BLOCK_JACOBI_MATRICES_H_ */
