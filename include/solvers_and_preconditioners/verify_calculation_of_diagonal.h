/*
 * VerifyCalculationOfDiagonal.h
 *
 *  Created on: Dec 1, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_VERIFY_CALCULATION_OF_DIAGONAL_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_VERIFY_CALCULATION_OF_DIAGONAL_H_

#include <deal.II/lac/parallel_vector.h>

/*
 *  To check the correctness of the efficient computation of the diagonal
 *  the result is compared to a naive calculation that simply applies the
 *  whole matrix-vector product N_dofs times. Accordingly, to call this
 *  function the Operator passed to this function has to implement a
 *  function called apply() that calculates the matrix-vector product.
 */
template<typename Operator, typename value_type>
void verify_calculation_of_diagonal(Operator                                  &op,
                                    parallel::distributed::Vector<value_type> &diagonal)
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
      ExcMessage("Number of MPI processes has to be 1."));

  ConditionalOStream pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
  pcout << "Verify calculation of diagonal:"<<std::endl;

  parallel::distributed::Vector<value_type>  diagonal_check(diagonal);
  parallel::distributed::Vector<value_type>  src(diagonal);
  parallel::distributed::Vector<value_type>  dst(diagonal);

  diagonal_check = 0.0; src = 0.0; dst = 0.0;

  /*
   *  Set dof-value i to 1.0, calculate matrix-vector
   *  product and store row i of the result in diagonal_check.
   */
  for (unsigned int i=0;i<diagonal.local_size();++i)
  {
    src.local_element(i) = 1.0;
    op.vmult(dst,src);
    diagonal_check.local_element(i) = dst.local_element(i);
    src.local_element(i) = 0.0;
  }

  value_type norm_diagonal = diagonal.l2_norm();
  value_type norm_diagonal_check = diagonal_check.l2_norm();

  pcout << std::endl
        << "L2 norm diagonal - Variant 1: " << std::setprecision(10)
        << norm_diagonal << std::endl;
  pcout << "L2 norm diagonal - Variant 2: " << std::setprecision(10)
        << norm_diagonal_check << std::endl;

  diagonal_check.add(-1.0,diagonal);
  value_type norm_error = diagonal_check.l2_norm();

  pcout << "L2 error diagonal: " << std::setprecision(10)
        << norm_error << std::endl <<std::endl;
}


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_VERIFY_CALCULATION_OF_DIAGONAL_H_ */
