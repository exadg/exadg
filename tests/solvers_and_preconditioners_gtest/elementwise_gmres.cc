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
#include <gtest/gtest.h>

#include "./test_utilities.h"

#include <exadg/solvers_and_preconditioners/preconditioners/elementwise_preconditioners.h>
#include <exadg/solvers_and_preconditioners/solvers/elementwise_krylov_solvers.h>

namespace
{
template<typename Number>
class ElementwiseGMRESTest : public ::testing::Test
{
protected:
  double const tol = 1.0e-14;

  using Matrix = TestUtilities::MyMatrix<Number>;
  using Vector = TestUtilities::MyVector<Number>;

  using Preconditioner = ExaDG::Elementwise::PreconditionerIdentity<Number>;
  using Solver         = ExaDG::Elementwise::SolverGMRES<Number, Matrix, Preconditioner>;

  using ::testing::Test::SetUp; // silence -Woverloaded-virtual warning
  void
  SetUp(unsigned int system_size)
  {
    gmres_solver   = std::make_unique<Solver>(system_size, ExaDG::SolverData(100, tol, tol, 30));
    preconditioner = std::make_unique<Preconditioner>(system_size);
  }

  static Vector
  calculate_residual(unsigned int const system_size, Matrix const & A, Vector & x, Vector & b)
  {
    Vector residual(system_size);
    A.vmult(residual.ptr(), x.ptr());
    residual.sadd(-1.0, b.ptr());

    return residual;
  }

  std::unique_ptr<Solver>         gmres_solver;
  std::unique_ptr<Preconditioner> preconditioner;
};

/*
 * Tests using double type
 */
using ElementwiseGMRESTestDouble = ElementwiseGMRESTest<double>;

TEST_F(ElementwiseGMRESTestDouble, SolvesSmallSystem)
{
  unsigned int const system_size = 3;
  SetUp(system_size);

  Vector b(system_size);
  b.set_value(1.0, 0);
  b.set_value(4.0, 1);
  b.set_value(6.0, 2);

  Vector x(system_size);
  x.init();

  Matrix A(system_size);
  A.set_value(1.0, 0, 0);
  A.set_value(2.0, 0, 1);
  A.set_value(3.0, 0, 2);
  A.set_value(2.0, 1, 0);
  A.set_value(3.0, 1, 1);
  A.set_value(1.0, 1, 2);
  A.set_value(3.0, 2, 0);
  A.set_value(1.0, 2, 1);
  A.set_value(2.0, 2, 2);

  Vector res_initial = calculate_residual(system_size, A, x, b);
  gmres_solver->solve(&A, x.ptr(), b.ptr(), preconditioner.get());
  Vector res_final = calculate_residual(system_size, A, x, b);

  EXPECT_NEAR(7.3e+00, res_initial.l2_norm(), 1.0e-01);
  EXPECT_NEAR(3.6e-15, res_final.l2_norm(), 1.0e-16);
}

TEST_F(ElementwiseGMRESTestDouble, SolverLargeSystem)
{
  unsigned int const system_size = 10000;
  SetUp(system_size);

  Vector b(system_size);
  for(unsigned int i = 0; i < system_size; ++i)
    b.set_value(1.0, i);

  Vector x(system_size);
  x.init();

  Matrix A(system_size);
  for(unsigned int i = 0; i < system_size; ++i)
  {
    A.set_value(2.0, i, i);
    if(i > 1)
    {
      A.set_value(-1.0, i - 1, i);
      A.set_value(-1.0, i, i - i);
    }
  }

  Vector res_initial = calculate_residual(system_size, A, x, b);
  gmres_solver->solve(&A, x.ptr(), b.ptr(), preconditioner.get());
  Vector res_final = calculate_residual(system_size, A, x, b);

  EXPECT_NEAR(1.0e+02, res_initial.l2_norm(), 1.0e+01);
  EXPECT_NEAR(5.9e-13, res_final.l2_norm(), tol);
}

/*
 * Tests using dealii::VectorizedArray<double> type
 */
using ElementwiseGMRESTestVectorizedDouble = ElementwiseGMRESTest<dealii::VectorizedArray<double>>;

TEST_F(ElementwiseGMRESTestVectorizedDouble, ConvergesFromExactSolution)
{
  unsigned int const system_size = 3;
  SetUp(system_size);

  Vector b(system_size);
  b.set_value(1.0, 0);
  b.set_value(2.0, 1);
  b.set_value(3.0, 2);

  Vector x(system_size);
  x.init();
  x.set_value(1.0, 0);
  x.set_value(1.0, 1);
  x.set_value(1.0, 2);

  Matrix A(system_size);
  A.set_value(1.0, 0, 0);
  A.set_value(2.0, 1, 1);
  A.set_value(3.0, 2, 2);

  gmres_solver->solve(&A, x.ptr(), b.ptr(), preconditioner.get());
  Vector res = calculate_residual(system_size, A, x, b);

  dealii::VectorizedArray<double> const l2_norm = res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    EXPECT_LT(l2_norm[v], 1.0e-12);
}

TEST_F(ElementwiseGMRESTestVectorizedDouble, ConvergesFromZero)
{
  unsigned int const system_size = 3;
  SetUp(system_size);

  Vector b(system_size);
  b.set_value(1.0, 0);
  b.set_value(4.0, 1);
  b.set_value(6.0, 2);

  Vector x(system_size);
  x.init();

  Matrix A(system_size);
  A.set_value(1.0, 0, 0);
  A.set_value(2.0, 0, 1);
  A.set_value(3.0, 0, 2);
  A.set_value(2.0, 1, 0);
  A.set_value(3.0, 1, 1);
  A.set_value(1.0, 1, 2);
  A.set_value(3.0, 2, 0);
  A.set_value(1.0, 2, 1);
  A.set_value(2.0, 2, 2);

  gmres_solver->solve(&A, x.ptr(), b.ptr(), preconditioner.get());
  Vector res = calculate_residual(system_size, A, x, b);

  dealii::VectorizedArray<double> l2_norm = res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    EXPECT_LT(l2_norm[v], 1.0e-12);
}

TEST_F(ElementwiseGMRESTestVectorizedDouble, SolvesDifferentEquations)
{
  unsigned int const system_size = 3;
  SetUp(system_size);

  Vector b(system_size);
  b.set_value(1.0, 0);
  b.set_value(2.0, 1);
  b.set_value(3.0, 2);

  Vector x(system_size);
  x.init();
  x.set_value(1.0, 0);
  x.set_value(1.0, 1);

  dealii::VectorizedArray<double> inhom_array(1.0);
  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    if(v > 1)
      inhom_array[v] = 0.0;

  x.set_value(inhom_array, 2);

  Matrix A(system_size);
  A.set_value(1.0, 0, 0);
  A.set_value(2.0, 1, 1);
  A.set_value(3.0, 2, 2);

  gmres_solver->solve(&A, x.ptr(), b.ptr(), preconditioner.get());
  Vector res = calculate_residual(system_size, A, x, b);

  dealii::VectorizedArray<double> l2_norm = res.l2_norm();

  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
    EXPECT_LT(l2_norm[v], 1.0e-12);
}
} // namespace