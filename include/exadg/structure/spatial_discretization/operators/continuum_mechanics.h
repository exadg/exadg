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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_
#define EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/physics/transformations.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number, typename TensorType>
inline DEAL_II_ALWAYS_INLINE //
  TensorType
  add_identity(TensorType tensor)
{
  for(unsigned int i = 0; i < dim; i++)
  {
    tensor[i][i] = tensor[i][i] + 1.0;
  }

  return tensor;
}

template<int dim, typename Number, typename TensorType>
inline DEAL_II_ALWAYS_INLINE //
  TensorType
  subtract_identity(TensorType tensor)
{
  for(unsigned int i = 0; i < dim; i++)
  {
    tensor[i][i] = tensor[i][i] - 1.0;
  }

  return tensor;
}

// Create a symmetric tensor from a tensor H plus H^T.
// Note that we have (H + H^T)^T = H^T + (H^T)^T = H^T + H = H + H^T.
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_H_plus_HT(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & H)
{
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> result;

  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      result[i][j] = H[i][j] + H[j][i];
    }
  }

  // Debug output.
  if constexpr(false)
  {
    if constexpr(std::is_same_v<Number, double>)
    {
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> check   = H + transpose(H);
      dealii::VectorizedArray<Number>                         sum_err = 0.0;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          sum_err += std::abs(check[i][j] - result[i][j]);
        }
      }
      std::cout << "compute_H_plus_HT : sum_err = " << sum_err << "\n";
      AssertThrow(sum_err.sum() < 1e-18,
                  dealii::ExcMessage("Check in `compute_H_plus_HT()` failed."));
    }
  }

  return result;
}

// Create a symmetric tensor from a tensor H^T times H.
// Note that we have (H^T * H)^T = H^T * (H^T)^T = H^T * H.
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_HT_times_H(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & H)
{
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> result;

  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      for(unsigned int k = 0; k < dim; ++k)
      {
        result[i][j] += /* HT[i][k] */ H[k][i] * H[k][j];
      }
    }
  }

  // Debug output.
  if constexpr(false)
  {
    if constexpr(std::is_same_v<Number, double>)
    {
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> check   = transpose(H) * H;
      dealii::VectorizedArray<Number>                         sum_err = 0.0;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          sum_err += std::abs(check[i][j] - result[i][j]);
        }
      }
      std::cout << "compute_HT_times_H : sum_err = " << sum_err << "\n";
      AssertThrow(sum_err.sum() < 1e-18,
                  dealii::ExcMessage("Check in `compute_HT_times_H()` failed."));
    }
  }

  return result;
}

// Create a symmetric tensor from a tensor H times H^T.
// Note that we have (H * H^T)^T = (H^T)^T * H^T = H * H^T.
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_H_times_HT(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & H)
{
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> result;

  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      for(unsigned int k = 0; k < dim; ++k)
      {
        result[i][j] += H[i][k] * H[j][k] /* HT[k][j] */;
      }
    }
  }

  // Debug output.
  if constexpr(false)
  {
    if constexpr(std::is_same_v<Number, double>)
    {
      dealii::VectorizedArray<Number> sum_err = 0.0;

      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> check = H * transpose(H);
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          sum_err += std::abs(check[i][j] - result[i][j]);
        }
      }
      std::cout << "compute_H_times_HT : sum_err = " << sum_err << "\n";
      AssertThrow(sum_err.sum() < 1e-18,
                  dealii::ExcMessage("Check in `compute_H_times_HT()` failed."));
    }
  }

  return result;
}

// Compute the push-forward of a symmetric tensor `S`, i.e.,
// tau = (1/J) * F * S * F^T,
// with `S` being symmetric, and therefore symmetric `tau`.
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_push_forward(dealii::VectorizedArray<Number> const &                                  J,
                       dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> const & S,
                       dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const &          F)
{
  // Compute the non-symmetric S * F^T, since we do not want to recompute
  // the components for the triple matrix product.
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> S_times_FT;
  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = 0; j < dim; ++j)
    {
      for(unsigned int k = 0; k < dim; ++k)
      {
        S_times_FT[i][j] += S[i][k] * F[j][k] /* FT[k][j] */;
      }
    }
  }

  // Compute the remaining product, exploit symmetry of the result.
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> result;
  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      for(unsigned int k = 0; k < dim; ++k)
      {
        result[i][j] += F[i][k] * S_times_FT[k][j];
      }
    }
  }

  // Perform scaling with 1/J.
  result /= J;

  // Debug output.
  if constexpr(false)
  {
    if constexpr(std::is_same_v<Number, double>)
    {
      dealii::VectorizedArray<Number> sum_err = 0.0;

      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> check =
        (1.0 / J) * F * S * transpose(F);
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          sum_err += std::abs(check[i][j] - result[i][j]);
        }
      }
      std::cout << "compute_push_forward : sum_err = " << sum_err << "\n";
      AssertThrow(sum_err.sum() < 1e-18,
                  dealii::ExcMessage("Check in `compute_push_forward()` failed."));
    }
  }

  return result;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_F(const dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & gradient_displacement)
{
  return add_identity<dim, Number>(gradient_displacement);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  get_E(const dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & gradient_displacement)
{
  // E = 0.5 (F^T * F) = 0.5 * (H + H^T + H^T * H),
  // where F is the deformation gradient and H is `gradient_displacement`.
  // Note that this version has also improved numerical stability.
  return (0.5 *
          (compute_H_plus_HT(gradient_displacement) + compute_HT_times_H(gradient_displacement)));
}

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
