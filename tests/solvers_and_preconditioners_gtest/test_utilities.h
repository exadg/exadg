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

#ifndef EXADG_TESTS_HELPER_LINEAR_ALGEBRA_H
#define EXADG_TESTS_HELPER_LINEAR_ALGEBRA_H

#include <deal.II/base/aligned_vector.h>

namespace TestUtilities
{
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
    AssertThrow(i < M && j < M, dealii::ExcMessage("Index exceeds matrix dimensions."));

    data[i * M + j] = value;
  }

private:
  // number of rows and columns of matrix
  unsigned int const                M;
  dealii::AlignedVector<value_type> data;
};

} // namespace TestUtilities

#endif // EXADG_TESTS_HELPER_LINEAR_ALGEBRA_H
