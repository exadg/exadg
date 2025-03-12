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

/**************************************************************************************/
/*                                                                                    */
/*                                        HEADER                                      */
/*                                                                                    */
/**************************************************************************************/

// C++
#include <fstream>
#include <iostream>
#include <sstream>

// deal.II
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/elementwise_preconditioners.h>
#include <exadg/solvers_and_preconditioners/solvers/elementwise_krylov_solvers.h>

namespace ExaDG
{
/**************************************************************************************/
/*                                                                                    */
/*                                   PARAMETERS                                       */
/*                                                                                    */
/**************************************************************************************/
unsigned int const M = 3;

double const tol = 1.e-14;

/*
 * Own implementation of vector class.
 */
template<typename value_type>
class MyVector
{
public:
  MyVector(unsigned int const size) : M(size)
  {
    data.resize(M);
  }

  value_type *
  ptr()
  {
    return &data[0];
  }

  void
  init()
  {
    for(unsigned int i = 0; i < M; ++i)
      data[i] = value_type();
  }

  void
  set_value(value_type const value, unsigned int const i)
  {
    AssertThrow(i < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    data[i] = value;
  }

  void
  sadd(value_type factor, value_type * src)
  {
    for(unsigned int i = 0; i < M; ++i)
      data[i] += factor * src[i];
  }

  value_type
  l2_norm()
  {
    value_type l2_norm = value_type();

    for(unsigned int i = 0; i < M; ++i)
      l2_norm += data[i] * data[i];

    l2_norm = std::sqrt(l2_norm);

    return l2_norm;
  }

private:
  // number of rows and columns of matrix
  unsigned int const                M;
  dealii::AlignedVector<value_type> data;
};


/*
 * Own implementation of matrix class.
 */
template<typename value_type>
class MyMatrix
{
public:
  // Constructor.
  MyMatrix(unsigned int const size) : M(size)
  {
    data.resize(M * M);
  }

  void
  vmult(value_type * dst, value_type * src) const
  {
    for(unsigned int i = 0; i < M; ++i)
    {
      dst[i] = value_type();
      for(unsigned int j = 0; j < M; ++j)
        dst[i] += data[i * M + j] * src[j];
    }
  }

  void
  precondition(value_type * dst, value_type * src) const
  {
    // no preconditioner
    for(unsigned int i = 0; i < M; ++i)
    {
      dst[i] = src[i]; // /data[i*M+i];
    }
  }

  void
  init()
  {
    for(unsigned int i = 0; i < M; ++i)
      for(unsigned int j = 0; j < M; ++j)
        data[i * M + j] = value_type(0.0);
  }

  void
  set_value(value_type const value, unsigned int const i, unsigned int const j)
  {
    AssertThrow(i < M and j < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    data[i * M + j] = value;
  }

private:
  // number of rows and columns of matrix
  unsigned int const                M;
  dealii::AlignedVector<value_type> data;
};


/**************************************************************************************/
/*                                                                                    */
/*                                         MAIN                                       */
/*                                                                                    */
/**************************************************************************************/

// double
void
gmres_test_1a()
{
  std::cout << std::endl << "GMRES solver (double), size M=3:" << std::endl << std::endl;

  SolverData solver_data(100, tol, tol, 30);

  typedef Elementwise::PreconditionerIdentity<double>      Preconditioner;
  typedef MyMatrix<double>                                 Matrix;
  Preconditioner                                           preconditioner(M);
  Elementwise::SolverGMRES<double, Matrix, Preconditioner> gmres_solver(M, solver_data);

  MyVector<double> b(M);
  b.set_value(1.0, 0);
  b.set_value(4.0, 1);
  b.set_value(6.0, 2);

  MyVector<double> x(M);
  x.init();

  Matrix matrix(M);
  matrix.set_value(1.0, 0, 0);
  matrix.set_value(2.0, 0, 1);
  matrix.set_value(3.0, 0, 2);
  matrix.set_value(2.0, 1, 0);
  matrix.set_value(3.0, 1, 1);
  matrix.set_value(1.0, 1, 2);
  matrix.set_value(3.0, 2, 0);
  matrix.set_value(1.0, 2, 1);
  matrix.set_value(2.0, 2, 2);

  MyVector<double> resi(M);
  matrix.vmult(resi.ptr(), x.ptr());
  resi.sadd(-1.0, b.ptr());

  std::cout << "L2 norm of initial residual = " << std::scientific << std::setprecision(1)
            << resi.l2_norm() << std::endl;

  gmres_solver.solve(&matrix, x.ptr(), b.ptr(), &preconditioner);

  MyVector<double> res(M);
  matrix.vmult(res.ptr(), x.ptr());
  res.sadd(-1.0, b.ptr());

  std::cout << "L2 norm of final residual = " << std::scientific << std::setprecision(1)
            << res.l2_norm() << std::endl;
}

// double (larger system)
void
gmres_test_1b()
{
  std::cout << std::endl << "GMRES solver (double), size M=10000:" << std::endl << std::endl;

  unsigned int const M_large = 10000;
  SolverData         solver_data(100, tol, tol, 30);

  typedef Elementwise::PreconditionerIdentity<double>      Preconditioner;
  typedef MyMatrix<double>                                 Matrix;
  Preconditioner                                           preconditioner(M_large);
  Elementwise::SolverGMRES<double, Matrix, Preconditioner> gmres_solver(M_large, solver_data);

  MyVector<double> b(M_large);
  for(unsigned int i = 0; i < M_large; ++i)
    b.set_value(1.0, i);

  MyVector<double> x(M_large);
  x.init();

  Matrix matrix(M_large);

  // use this parameter to switch between two different matrices
  bool matrix_type = true;
  if(matrix_type)
  {
    for(unsigned int i = 0; i < M_large; ++i)
    {
      matrix.set_value(2.0, i, i);
      if(i > 1)
      {
        matrix.set_value(-1.0, i - 1, i);
        matrix.set_value(-1.0, i, i - i);
      }
    }
  }
  else
  {
    for(unsigned int i = 0; i < M_large; ++i)
    {
      for(unsigned int j = 0; j < M_large; ++j)
      {
        matrix.set_value((i + j) % 17, i, j);
      }
    }
  }

  MyVector<double> resi(M_large);
  matrix.vmult(resi.ptr(), x.ptr());
  resi.sadd(-1.0, b.ptr());

  std::cout << "L2 norm of initial residual = " << std::scientific << std::setprecision(1)
            << resi.l2_norm() << std::endl;

  gmres_solver.solve(&matrix, x.ptr(), b.ptr(), &preconditioner);

  MyVector<double> res(M_large);
  matrix.vmult(res.ptr(), x.ptr());
  res.sadd(-1.0, b.ptr());

  std::cout << "L2 norm of final residual = " << std::scientific << std::setprecision(1)
            << res.l2_norm() << std::endl;
}

// Vectorized Array: start with exact solution
void
gmres_test_2a()
{
  std::cout << std::endl
            << "GMRES solver (VectorizedArray<double>), size M=3, start with exact solution:"
            << std::endl
            << std::endl;

  SolverData solver_data(100, tol, tol, 30);

  typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<double>> Preconditioner;
  typedef MyMatrix<dealii::VectorizedArray<double>>                            Matrix;
  Preconditioner                                                               preconditioner(M);
  Elementwise::SolverGMRES<dealii::VectorizedArray<double>, Matrix, Preconditioner> gmres_solver(
    M, solver_data);

  MyVector<dealii::VectorizedArray<double>> b(M);
  b.set_value(dealii::make_vectorized_array<double>(1.0), 0);
  b.set_value(dealii::make_vectorized_array<double>(2.0), 1);
  b.set_value(dealii::make_vectorized_array<double>(3.0), 2);

  MyVector<dealii::VectorizedArray<double>> x(M);
  x.init();
  x.set_value(dealii::make_vectorized_array<double>(1.0), 0);
  x.set_value(dealii::make_vectorized_array<double>(1.0), 1);
  x.set_value(dealii::make_vectorized_array<double>(1.0), 2);

  Matrix A(M);
  A.set_value(dealii::make_vectorized_array<double>(1.0), 0, 0);
  A.set_value(dealii::make_vectorized_array<double>(2.0), 1, 1);
  A.set_value(dealii::make_vectorized_array<double>(3.0), 2, 2);

  /*
  MyVector<dealii::VectorizedArray<double>> initial_res(M);
  A.vmult(initial_res.ptr(), x.ptr());
  initial_res.sadd(dealii::make_vectorized_array<double>(-1.0), b.ptr());

  dealii::VectorizedArray<double> l2_norm_initial_res = initial_res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    std::cout << "L2 norm of initial residual[" << v << "] = " << l2_norm_initial_res[v] <<
  std::endl;
   */

  gmres_solver.solve(&A, x.ptr(), b.ptr(), &preconditioner);

  MyVector<dealii::VectorizedArray<double>> res(M);
  A.vmult(res.ptr(), x.ptr());
  res.sadd(dealii::make_vectorized_array<double>(-1.0), b.ptr());

  dealii::VectorizedArray<double> l2_norm = res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    AssertThrow(l2_norm[v] < 1.e-12, dealii::ExcMessage("Did not converge."));

  std::cout << "converged." << std::endl;
}

// dealii::VectorizedArray: start with zero solution
void
gmres_test_2b()
{
  std::cout << std::endl
            << "GMRES solver (VectorizedArray<double>), size M=3, start with zero solution:"
            << std::endl
            << std::endl;

  SolverData solver_data(100, 1e-12, 1e-12, 30);

  typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<double>> Preconditioner;
  typedef MyMatrix<dealii::VectorizedArray<double>>                            Matrix;
  Preconditioner                                                               preconditioner(M);
  Elementwise::SolverGMRES<dealii::VectorizedArray<double>, Matrix, Preconditioner> gmres_solver(
    M, solver_data);

  MyVector<dealii::VectorizedArray<double>> b(M);
  b.set_value(dealii::make_vectorized_array<double>(1.0), 0);
  b.set_value(dealii::make_vectorized_array<double>(4.0), 1);
  b.set_value(dealii::make_vectorized_array<double>(6.0), 2);

  MyVector<dealii::VectorizedArray<double>> x(M);
  x.init();

  Matrix A(M);
  A.set_value(dealii::make_vectorized_array<double>(1.0), 0, 0);
  A.set_value(dealii::make_vectorized_array<double>(2.0), 0, 1);
  A.set_value(dealii::make_vectorized_array<double>(3.0), 0, 2);
  A.set_value(dealii::make_vectorized_array<double>(2.0), 1, 0);
  A.set_value(dealii::make_vectorized_array<double>(3.0), 1, 1);
  A.set_value(dealii::make_vectorized_array<double>(1.0), 1, 2);
  A.set_value(dealii::make_vectorized_array<double>(3.0), 2, 0);
  A.set_value(dealii::make_vectorized_array<double>(1.0), 2, 1);
  A.set_value(dealii::make_vectorized_array<double>(2.0), 2, 2);

  /*
  MyVector<dealii::VectorizedArray<double>> initial_res(M);
  A.vmult(initial_res.ptr(), x.ptr());
  initial_res.sadd(dealii::make_vectorized_array<double>(-1.0), b.ptr());

  dealii::VectorizedArray<double> l2_norm_initial_res = initial_res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    std::cout << "L2 norm of initial residual[" << v << "] = " << l2_norm_initial_res[v] <<
  std::endl;
  */

  gmres_solver.solve(&A, x.ptr(), b.ptr(), &preconditioner);

  MyVector<dealii::VectorizedArray<double>> res(M);
  A.vmult(res.ptr(), x.ptr());
  res.sadd(dealii::make_vectorized_array<double>(-1.0), b.ptr());

  dealii::VectorizedArray<double> l2_norm = res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    AssertThrow(l2_norm[v] < 1.e-12, dealii::ExcMessage("Did not converge."));

  std::cout << "converged." << std::endl;
}

// dealii::VectorizedArray: solve different systems of equations for the
// different components of the vectorized array
void
gmres_test_2c()
{
  std::cout << std::endl
            << "GMRES solver (VectorizedArray<double>), size M=3, solve different systems:"
            << std::endl
            << std::endl;

  SolverData solver_data(100, tol, tol, 30);

  typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<double>> Preconditioner;
  typedef MyMatrix<dealii::VectorizedArray<double>>                            Matrix;
  Preconditioner                                                               preconditioner(M);
  Elementwise::SolverGMRES<dealii::VectorizedArray<double>, Matrix, Preconditioner> gmres_solver(
    M, solver_data);

  MyVector<dealii::VectorizedArray<double>> b(M);
  b.set_value(dealii::make_vectorized_array<double>(1.0), 0);
  b.set_value(dealii::make_vectorized_array<double>(2.0), 1);
  b.set_value(dealii::make_vectorized_array<double>(3.0), 2);

  MyVector<dealii::VectorizedArray<double>> x(M);
  x.init();
  x.set_value(dealii::make_vectorized_array<double>(1.0), 0);
  x.set_value(dealii::make_vectorized_array<double>(1.0), 1);

  dealii::VectorizedArray<double> inhom_array = dealii::make_vectorized_array<double>(1.0);
  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    if(v > 1)
      inhom_array[v] = 0.0;

  x.set_value(inhom_array, 2);

  Matrix A(M);
  A.set_value(dealii::make_vectorized_array<double>(1.0), 0, 0);
  A.set_value(dealii::make_vectorized_array<double>(2.0), 1, 1);
  A.set_value(dealii::make_vectorized_array<double>(3.0), 2, 2);

  /*
  MyVector<dealii::VectorizedArray<double>> initial_res(M);
  A.vmult(initial_res.ptr(), x.ptr());
  initial_res.sadd(dealii::make_vectorized_array<double>(-1.0), b.ptr());

  dealii::VectorizedArray<double> l2_norm_initial_res = initial_res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    std::cout << "L2 norm of initial residual[" << v << "] = " << l2_norm_initial_res[v] <<
  std::endl;
   */

  gmres_solver.solve(&A, x.ptr(), b.ptr(), &preconditioner);

  MyVector<dealii::VectorizedArray<double>> res(M);
  A.vmult(res.ptr(), x.ptr());
  res.sadd(dealii::make_vectorized_array<double>(-1.0), b.ptr());

  dealii::VectorizedArray<double> l2_norm = res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    AssertThrow(l2_norm[v] < 1.e-12, dealii::ExcMessage("Did not converge."));

  std::cout << "converged." << std::endl;
}

} // namespace ExaDG

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    dealii::deallog.depth_console(0);

    // double
    ExaDG::gmres_test_1a();
    ExaDG::gmres_test_1b();

    // dealii::VectorizedArray<double>
    ExaDG::gmres_test_2a();
    ExaDG::gmres_test_2b();
    ExaDG::gmres_test_2c();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
