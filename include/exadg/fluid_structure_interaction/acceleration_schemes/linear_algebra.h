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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_LINEAR_ALGEBRA_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_LINEAR_ALGEBRA_H_

// C/C++
#include <vector>

// deal.II
#include <deal.II/base/exceptions.h>

namespace ExaDG
{
namespace FSI
{
/*
 * Own implementation of matrix class.
 */
template<typename Number>
class Matrix
{
public:
  // Constructor.
  Matrix(unsigned int const size) : M(size)
  {
    data.resize(M * M);

    init();
  }

  void
  init()
  {
    for(unsigned int i = 0; i < M; ++i)
      for(unsigned int j = 0; j < M; ++j)
        data[i * M + j] = Number(0.0);
  }

  Number
  get(unsigned int const i, unsigned int const j) const
  {
    AssertThrow(i < M and j < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    return data[i * M + j];
  }

  void
  set(Number const value, unsigned int const i, unsigned int const j)
  {
    AssertThrow(i < M and j < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    data[i * M + j] = value;
  }

private:
  // number of rows and columns of matrix
  unsigned int const  M;
  std::vector<Number> data;
};

template<typename VectorType, typename Number>
void
compute_QR_decomposition(std::vector<VectorType> & Q, Matrix<Number> & R, Number const eps = 1.e-2)
{
  for(unsigned int i = 0; i < Q.size(); ++i)
  {
    Number const norm_initial = Number(Q[i].l2_norm());

    // orthogonalize
    for(unsigned int j = 0; j < i; ++j)
    {
      Number r_ji = Q[j] * Q[i];
      R.set(r_ji, j, i);
      Q[i].add(-r_ji, Q[j]);
    }

    // normalize or drop if linear dependent
    Number r_ii = Number(Q[i].l2_norm());
    if(r_ii < eps * norm_initial)
    {
      Q[i] = 0.0;
      for(unsigned int j = 0; j < i; ++j)
        R.set(0.0, j, i);
      R.set(1.0, i, i);
    }
    else
    {
      R.set(r_ii, i, i);
      Q[i] *= 1. / r_ii;
    }
  }
}

/*
 *  Matrix has to be upper triangular with d_ii != 0 for all 0 <= i < n
 */
template<typename Number>
void
backward_substitution(Matrix<Number> const &      matrix,
                      std::vector<Number> &       dst,
                      std::vector<Number> const & rhs)
{
  int const n = dst.size();

  for(int i = n - 1; i >= 0; --i)
  {
    double value = rhs[i];
    for(int j = i + 1; j < n; ++j)
    {
      value -= matrix.get(i, j) * dst[j];
    }

    dst[i] = value / matrix.get(i, i);
  }
}

template<typename Number, typename VectorType>
void
backward_substitution_multiple_rhs(Matrix<Number> const &          matrix,
                                   std::vector<VectorType> &       dst,
                                   std::vector<VectorType> const & rhs)
{
  int const n = dst.size();

  for(int i = n - 1; i >= 0; --i)
  {
    VectorType value = rhs[i];
    for(int j = i + 1; j < n; ++j)
    {
      value.add(-matrix.get(i, j), dst[j]);
    }

    dst[i].equ(1.0 / matrix.get(i, i), value);
  }
}

template<typename VectorType>
void
inv_jacobian_times_residual(VectorType &                                                  b,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & D_history,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & R_history,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & Z_history,
                            VectorType const &                                            residual)
{
  VectorType a = residual;

  // reset
  b = 0.0;

  for(int idx = Z_history.size() - 1; idx >= 0; --idx)
  {
    std::shared_ptr<std::vector<VectorType>> D = D_history[idx];
    std::shared_ptr<std::vector<VectorType>> R = R_history[idx];
    std::shared_ptr<std::vector<VectorType>> Z = Z_history[idx];

    int const           k = Z->size();
    std::vector<double> Z_times_a(k, 0.0);
    for(int i = 0; i < k; ++i)
      Z_times_a[i] = (*Z)[i] * a;

    // add to b
    for(int i = 0; i < k; ++i)
      b.add(Z_times_a[i], (*D)[i]);

    // add to a
    for(int i = 0; i < k; ++i)
      a.add(-Z_times_a[i], (*R)[i]);
  }
}

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_LINEAR_ALGEBRA_H_ */
