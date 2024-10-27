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

#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_

// clang-format off
#define ONE_THIRD   0.33333333333333333333
#define TWO_THIRDS  0.66666666666666666666
#define FOUR_THIRDS 1.33333333333333333333
#define ONE_NINTH   0.11111111111111111111
#define TWO_NINTHS  0.22222222222222222222
#define LOG2E       1.442695040888963387
// clang-format on

// C++
#include <tuple>

// deal.II
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/transformations.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  void
  bound_tensor(dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> & tensor,
               Number const &                                                     upper_bound)
{
  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      tensor[i][j] = std::min(tensor[i][j], dealii::make_vectorized_array(upper_bound));
    }
  }
}

// Compute f(x) = log(1 + x)
template<typename Number, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  fast_approx_log1p(dealii::VectorizedArray<Number> const & x)
{
  unsigned int constexpr option = 1;

  if constexpr(option == 0)
  {
    Number values[dealii::VectorizedArray<Number>::size()];
    for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    {
      if constexpr(stable_formulation)
      {
        values[i] = std::log1p(x[i]);
      }
      else
      {
        values[i] = std::log(1.0 + x[i]);
      }
    }

    dealii::VectorizedArray<Number> out;
    out.load(&values[0]);
    return out;
  }
  else if constexpr(option == 1)
  {
    // Use Taylor expansions proposed by
    // Shakeri et al. [https://arxiv.org/pdf/2401.13196]
    // log1p(x) = log(1 + x) = 2 * sum_(n = 0 to inf) (x / (2 + x))^(2*n+1) / (2*n+1)
    // Note that the point of evolution is x = 0, since we aim to evaluate
    // log1p(Jm1) = log(1 + J - 1) = log(J) ~= log1p(0)
    unsigned int constexpr n_terms = 6;
    Number two_over_2np1[n_terms];
    for(unsigned int i = 0; i < n_terms; ++i)
    {
      two_over_2np1[i] = 2.0 / static_cast<Number>(2 * i + 1);
    }

    dealii::VectorizedArray<Number> const x_over_2px = x / (2.0 + x);
    dealii::VectorizedArray<Number>       last_power_of_x_over_2px(static_cast<Number>(1.0));

    // First iteration = initialization.
    dealii::VectorizedArray<Number> out = two_over_2np1[0] * x_over_2px;
    for(unsigned int i = 1; i < n_terms; ++i)
    {
      last_power_of_x_over_2px *= x_over_2px * x_over_2px;
      out += two_over_2np1[i] * last_power_of_x_over_2px * x_over_2px;
    }

    // The argument x in log1p(x) = log(x+1) has to be larger than -1.
    // out = x > x_limit ? out : -inf
    out = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      x,
      dealii::make_vectorized_array<Number>(static_cast<Number>(-1.0)),
      out,
      dealii::make_vectorized_array<Number>(-std::numeric_limits<Number>::infinity()));

    if constexpr(false) // test for doubles and float
    {
      Number max_rel_err = 0.0;
      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        max_rel_err =
          std::max(max_rel_err, std::abs(out[i] - std::log1p(x[i])) / std::abs(std::log1p(x[i])));
      }
      std::string const number_type_str = std::is_same_v<Number, double> ? "double" : "float";
      if(max_rel_err > 1e-5)
      {
        std::cout << "(0) max_rel_err = " << max_rel_err << " (" << number_type_str << ")\n";
      }
    }

    return out;
  }
  else if constexpr(option == 2)
  {
    // Use ideas from Proell et al. [https://arxiv.org/pdf/2402.17580].
    AssertThrow(option < 2, dealii::ExcMessage("This option has not been implemented yet."));
    return (dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN()));
  }
}

namespace Internal
{
// Coefficients for fast approximate `exp()` function, see
// Proell et al. [https://arxiv.org/pdf/2402.17580].
constexpr std::array<double, 8> COEFF_EXP = {{1.21307181188968e-10,
                                              0.30685281026575,
                                              -0.240226342399359,
                                              -0.0555053313414954,
                                              -0.0096135243288483,
                                              -0.00134288475963084,
                                              -0.000143131744483589,
                                              -2.1595656126349e-05}};

// Bitshift constants: 2^52, 2^52 * 1023, 2^23, 2^23 * 127.
constexpr double TWO_POW_52            = 4503599627370496;
constexpr double TWO_POW_52_TIMES_1023 = 4607182418800017408;
constexpr float  TWO_POW_23            = 8388608;
constexpr float  TWO_POW_23_TIMES_127  = 1065353216;

// Maximum argument for the fast_exp() function such that float representations do not overflow.
// This is 38/log10(e) for single precision, double precision allows up to 308/log10(e) for the
// argument.

constexpr double MAX_ABS_EXP_ARG_DOUBLE = 709.196208642;
constexpr float  MAX_ABS_EXP_ARG_SINGLE = 87.4982335338;

} // namespace Internal

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 512 && defined(__AVX512F__)
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<double, 8>
  floor(dealii::VectorizedArray<double, 8> const & in)
{
  dealii::VectorizedArray<double, 8> out;

  out.data = _mm512_roundscale_pd(in.data, _MM_FROUND_TO_NEG_INF);

  return out;
}

inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<float, 16>
  floor(dealii::VectorizedArray<float, 16> const & in)
{
  dealii::VectorizedArray<float, 16> out;

  out.data = _mm512_roundscale_ps(in.data, _MM_FROUND_TO_NEG_INF);

  return out;
}
#endif

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 256 && defined(__AVX__)
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<double, 4>
  floor(dealii::VectorizedArray<double, 4> const & in)
{
  dealii::VectorizedArray<double, 4> out;

  out.data = _mm256_floor_pd(in.data);

  return out;
}

inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<float, 8>
  floor(dealii::VectorizedArray<float, 8> const & in)
{
  dealii::VectorizedArray<float, 8> out;

  out.data = _mm256_floor_ps(in.data);

  return out;
}
#endif

template<typename VectorizedArrayType, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArrayType
  fma(Number const x, VectorizedArrayType const y, Number const z)
{
  VectorizedArrayType out = z;

  out += x * y;

  return out;
}

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 512 && defined(__AVX512F__)
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<double, 8>
  type_cast(dealii::VectorizedArray<double, 8> const & in)
{
  dealii::VectorizedArray<double, 8> out;

  __m512i integer = _mm512_cvt_roundpd_epi64(in.data, _MM_FROUND_NO_EXC);
  out.data        = _mm512_castsi512_pd(integer);

  return out;
}

inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<float, 16>
  type_cast(dealii::VectorizedArray<float, 16> const & in)
{
  dealii::VectorizedArray<float, 16> out;

  __m512i integer = _mm512_cvt_roundps_epi32(in.data, _MM_FROUND_NO_EXC);
  out.data        = _mm512_castsi512_ps(integer);

  return out;
}
#endif

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 256 && defined(__AVX__)
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<double, 4>
  type_cast(dealii::VectorizedArray<double, 4> const & in)
{
  dealii::VectorizedArray<double, 4> out;

  double double_values[4];
  in.store(&double_values[0]);
  int64_t int_values[4];
  for(unsigned int i = 0; i < 4; ++i)
  {
    int_values[i] = static_cast<int64_t>(double_values[i]);
  }
  out.data = _mm256_castsi256_pd(_mm256_loadu_si256((__m256i *)int_values));

  return out;
}

inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<float, 8>
  type_cast(dealii::VectorizedArray<float, 8> const & in)
{
  dealii::VectorizedArray<float, 8> out;

  float float_values[8];
  in.store(&float_values[0]);
  int32_t int_values[8];
  for(unsigned int i = 0; i < 8; ++i)
  {
    int_values[i] = static_cast<int32_t>(float_values[i]);
  }
  out.data = _mm256_castsi256_ps(_mm256_loadu_si256((__m256i *)int_values));

  return out;
}
#endif

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 128 && defined(__SSE2__)
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<double, 2>
  type_cast(dealii::VectorizedArray<double, 2> const & in)
{
  std::cout << "This has not been tested. (5)\n";

  dealii::VectorizedArray<double, 2> out;

  double double_values[2];
  in.store(&double_values[0]);
  int64_t int_values[2];
  for(unsigned int i = 0; i < 2; ++i)
  {
    int_values[i] = static_cast<int64_t>(double_values[i]);
  }
  out.data = _mm_castsi128_pd(_mm_loadu_si128((__m128i *)int_values));

  return out;
}

inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<float, 4>
  type_cast(dealii::VectorizedArray<float, 4> const & in)
{
  std::cout << "This has not been tested. (6)\n";

  dealii::VectorizedArray<float, 4> out;

  float float_values[4];
  in.store(&float_values[0]);
  int32_t int_values[4];
  for(unsigned int i = 0; i < 4; ++i)
  {
    int_values[i] = static_cast<int32_t>(float_values[i]);
  }
  out.data = _mm_castsi128_ps(_mm_loadu_si128((__m128i *)int_values));

  return out;
}
#endif

inline DEAL_II_ALWAYS_INLINE dealii::VectorizedArray<double, 1>
                             type_cast(dealii::VectorizedArray<double, 1> const & in)
{
  std::cout << "This has not been tested. (7)\n";

  dealii::VectorizedArray<double, 1> out;

  auto result = static_cast<int64_t>(in.data);
  std::memcpy(&out.data, &result, sizeof(out.data));

  return out;
}

inline DEAL_II_ALWAYS_INLINE dealii::VectorizedArray<float, 1>
                             type_cast(dealii::VectorizedArray<float, 1> const & in)
{
  std::cout << "This has not been tested. (8)\n";

  dealii::VectorizedArray<float, 1> out;

  auto result = static_cast<int32_t>(in.data);
  std::memcpy(&out.data, &result, sizeof(out.data));

  return out;
}

// Fast approximate `exp()` function, see
// Proell et al. [https://arxiv.org/pdf/2402.17580].
template<typename Number>
static constexpr inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  fast_approx_exp(dealii::VectorizedArray<Number> x)
{
  x *= LOG2E;

  dealii::VectorizedArray<Number> fractional_part = x - floor(x);

  bool constexpr use_8_else_4_terms = false;
  if constexpr(use_8_else_4_terms)
  {
    dealii::VectorizedArray<Number> fractional_part2 = fractional_part * fractional_part;
    dealii::VectorizedArray<Number> fractional_part4 = fractional_part2 * fractional_part2;
    x -= fma(fma(fma(Internal::COEFF_EXP[7], fractional_part, Internal::COEFF_EXP[6]),
                 fractional_part2,
                 fma(Internal::COEFF_EXP[5], fractional_part, Internal::COEFF_EXP[4])),
             fractional_part4,
             fma(fma(Internal::COEFF_EXP[3], fractional_part, Internal::COEFF_EXP[2]),
                 fractional_part2,
                 fma(Internal::COEFF_EXP[1], fractional_part, Internal::COEFF_EXP[0])));
  }
  else
  {
    dealii::VectorizedArray<Number> fractional_part2 = fractional_part * fractional_part;
    x -= fma(fma(Internal::COEFF_EXP[3], fractional_part, Internal::COEFF_EXP[2]),
             fractional_part2,
             fma(Internal::COEFF_EXP[1], fractional_part, Internal::COEFF_EXP[0]));
  }

  if constexpr(std::is_same_v<Number, double>)
  {
    return type_cast(Internal::TWO_POW_52 * x + Internal::TWO_POW_52_TIMES_1023);
  }
  else if constexpr(std::is_same_v<Number, float>)
  {
    return type_cast(Internal::TWO_POW_23 * x + Internal::TWO_POW_23_TIMES_127);
  }
  else
  {
    return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
  }
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  exp_limited(dealii::VectorizedArray<Number> const & x, Number const & upper_bound)
{
  bool constexpr option = 0;

  dealii::VectorizedArray<Number> out;
  if constexpr(option == 0)
  {
    out = fast_approx_exp(x);

    // Enforce the upper bound based on the single precision upper limit to the argument
    // such that the result would overflow single precision.
    // out = x < x_limit ? out : limit
    out = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      x,
      dealii::make_vectorized_array<Number>(static_cast<Number>(Internal::MAX_ABS_EXP_ARG_SINGLE)),
      out,
      dealii::make_vectorized_array<Number>(upper_bound));
  }
  else
  {
    for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    {
      out[i] = std::exp(x[i]);
    }
  }

  // Bound result with supplied upper limit.
  out = std::min(out, dealii::make_vectorized_array(upper_bound));

  if constexpr(false and option == 0) // test for doubles and float
  {
    Number max_rel_err = 0.0;
    for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    {
      Number const abs_exp_x = x[i] < static_cast<Number>(Internal::MAX_ABS_EXP_ARG_SINGLE) ?
                                 std::abs(std::exp(x[i])) :
                                 upper_bound;
      Number const limited_abs_exp_x = std::min(abs_exp_x, upper_bound);
      max_rel_err = std::max(max_rel_err, std::abs(limited_abs_exp_x - out[i]) / limited_abs_exp_x);
    }

    std::string const number_type_str = std::is_same_v<Number, double> ? "double" : "float";
    std::cout << "(9) max_rel_err = " << max_rel_err << " (" << number_type_str << ")\n";
  }

  return out;
}

// Compute f(x) = e^(1 + x)
template<typename Number, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  fast_approx_powp1(dealii::VectorizedArray<Number> const & x, Number const & e)
{
  unsigned int constexpr option = 0;

  if constexpr(option == 0)
  {
    // Just use pow().
    return pow(1.0 + x, e);
  }
  else if(option == 1)
  {
    // Loop over the vectorized array's entries:
    // (x + 1)^e = exp( e * log(x + 1)),
    // which is numerically unstable for x -> 0
    // or
    // (x + 1)^e = exp( e * logp1(x)),
    // which is numerically stable for x -> 0.
    Number values[dealii::VectorizedArray<Number>::size()];
    for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    {
      if constexpr(stable_formulation)
      {
        values[i] = std::exp(e * std::log1p(x[i]));
      }
      else
      {
        values[i] = std::exp(e * std::log(1.0 + x[i]));
      }
    }

    dealii::VectorizedArray<Number> out;
    out.load(&values[0]);
    return out;
  }
  else if(option == 2)
  {
    // Use Taylor expansions for exp(x) and log1p(x) similar to
    // Shakeri et al. [https://arxiv.org/pdf/2401.13196]
    AssertThrow(option < 2, dealii::ExcMessage("This option has not been implemented yet."));
    return (dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN()));
  }
  else if constexpr(option == 3)
  {
    // Use ideas from Proell et al. [https://arxiv.org/pdf/2402.17580].
    AssertThrow(option < 3, dealii::ExcMessage("This option has not been implemented yet."));
    return (dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN()));
  }
}

// Create a symmetric tensor from a tensor H plus H^T
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
      std::cout << "(10) sum_err = " << sum_err << "\n";
    }
  }

  return result;
}

// Create a symmetric tensor from a tensor H^T times H
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
      std::cout << "(11) sum_err = " << sum_err << "\n";
    }
  }

  return result;
}

// Create a symmetric tensor from a tensor H times H^T
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
      std::cout << "(12) sum_err = " << sum_err << "\n";
    }
  }

  return result;
}

// Compute the symmetric(!) product of two symmetric tensors:
// C = A * B, with A = A^T, B = B^T *and additionally* C = C^T
// Note that this is a special case, since in general, we have
// C^T = (A * B)^T = B^T * A = B * A != A * B.
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_symmetric_product(
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> const & A,
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> const & B)
{
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> result;

  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = i; j < dim; ++j)
    {
      for(unsigned int k = 0; k < dim; ++k)
      {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  if constexpr(false)
  {
    if constexpr(std::is_same_v<Number, double>)
    {
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> A_, B_;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          A_[i][j] = A[i][j];
          B_[i][j] = B[i][j];
        }
      }

      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> check;

      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          check[i][j] = A_[i][0] * B_[0][j];
          for(unsigned int k = 1; k < dim; ++k)
          {
            check[i][j] += A[i][k] * B[k][j];
          }
        }
      }

      dealii::VectorizedArray<Number> sum_err = 0.0, sym_err = 0.0, magn = 0.0;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          magn += std::abs(check[i][j]);
          sym_err += std::abs(A_[i][j] - A[j][i]) + std::abs(B_[i][j] - B[j][i]);
          sum_err += std::abs(check[i][j] - result[i][j]);
        }
      }
      std::cout << "(13) sum_err = " << sum_err << ", sym_err = " << sym_err
                << ", magn = " << magn / static_cast<Number>(dim * dim) << "\n";
    }
  }

  return result;
}

// Compute the push-forward of a symmetric tensor, i.e.,
// tau = F * S * F^T,
// with S being symmetric, and therefore symmetric tau.
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_push_forward(dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> const & S,
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

  if constexpr(false)
  {
    if constexpr(std::is_same_v<Number, double>)
    {
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> S_;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          S_[i][j] = S[i][j];
        }
      }

      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> check = F * S_ * transpose(F);

      dealii::VectorizedArray<Number> sum_err = 0.0;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          sum_err += std::abs(check[i][j] - result[i][j]);
        }
      }
      std::cout << "(14) sum_err = " << sum_err << "\n";
    }
  }

  return result;
}

template<int dim, typename Number, typename TypeScale>
inline DEAL_II_ALWAYS_INLINE //
  void
  add_scaled_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & tmp,
                      TypeScale const &                                         scale)
{
  for(unsigned int i = 0; i < dim; ++i)
  {
    tmp[i][i] = tmp[i][i] + scale;
  }
}

template<int dim, typename Number, typename TypeScale>
inline DEAL_II_ALWAYS_INLINE //
  void
  add_scaled_identity(dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> & tmp,
                      TypeScale const &                                                  scale)
{
  for(unsigned int i = 0; i < dim; ++i)
  {
    tmp[i][i] = tmp[i][i] + scale;
  }
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  compute_F(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> F = gradient_displacement;
  add_scaled_identity(F, static_cast<Number>(1.0));
  return F;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_C_inv(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & F_inv)
{
  return compute_H_times_HT<dim, Number>(F_inv);
}


template<int dim, typename Number, typename TypeScale, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_E_scaled(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    TypeScale const &                                               scale)
{
  if constexpr(stable_formulation)
  {
    // E = 0.5 * (H + H^T + H^T * H)
    // where H = gradient_displacement
    return ((0.5 * scale) *
            (compute_H_plus_HT(gradient_displacement) + compute_HT_times_H(gradient_displacement)));
  }
  else
  {
    // E = 0.5 * (C - I) = 0.5 * (F^T * F - I)
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> E =
      compute_HT_times_H(compute_F(gradient_displacement));
    add_scaled_identity(E, static_cast<Number>(-1.0));
    return (E * (0.5 * scale));
  }
}

template<int dim, typename Number, typename TypeScale, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  compute_e_scaled(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    TypeScale const &                                               scale)
{
  if constexpr(stable_formulation)
  {
    // e = 0.5 * (H + H^T + H*H^T)
    // where H = gradient_displacement
    return ((0.5 * scale) *
            (compute_H_plus_HT(gradient_displacement) + compute_H_times_HT(gradient_displacement)));
  }
  else
  {
    // e = 0.5 * (b - I)
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> e =
      compute_H_times_HT(compute_F(gradient_displacement));
    add_scaled_identity(e, static_cast<Number>(-1.0));
    return (e * (0.5 * scale));
  }
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_identity_tensor()
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> identity_tensor;
  add_scaled_identity(identity_tensor, static_cast<Number>(1.0));
  return identity_tensor;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  get_identity_symmetric_tensor()
{
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> identity_tensor;
  add_scaled_identity(identity_tensor, static_cast<Number>(1.0));
  return identity_tensor;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_zero_tensor()
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> zero_tensor;
  return zero_tensor;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
  get_zero_symmetric_tensor()
{
  dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> zero_tensor;
  return zero_tensor;
}

template<int dim, typename Number, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  compute_Jm1(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  if constexpr(stable_formulation)
  {
    // See [Shakeri et al., 2024, https://doi.org/10.48550/arXiv.2401.13196]
    if constexpr(dim == 2)
    {
      // clang-format off
      return (gradient_displacement[0][0] + gradient_displacement[1][1]
            + gradient_displacement[0][0] * gradient_displacement[1][1]
            - gradient_displacement[0][1] * gradient_displacement[1][0]);
      // clang-format on
    }
    else if constexpr(dim == 3)
    {
      // clang-format off
      // Sum terms of starting from lowest order of magnitude individually.
      dealii::VectorizedArray<Number> Jm1 = determinant(gradient_displacement);
      Jm1 += (  gradient_displacement[0][0] * gradient_displacement[1][1]
              + gradient_displacement[1][1] * gradient_displacement[2][2]
              + gradient_displacement[0][0] * gradient_displacement[2][2])
           - (  gradient_displacement[0][1] * gradient_displacement[1][0]
              + gradient_displacement[1][2] * gradient_displacement[2][1]
              + gradient_displacement[0][2] * gradient_displacement[2][0]);
      Jm1 += trace(gradient_displacement);
      return Jm1;
      // clang-format on
    }
    else
    {
      AssertThrow(dim != 2 and dim != 3,
                  dealii::ExcMessage("Unexpected dim. Choose dim == 2 or dim == 3."));
      return (dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN()));
    }
  }
  else
  {
    return (determinant(compute_F(gradient_displacement)) - 1.0);
  }
}

// Compute J^2-1 in a numerically stable manner, which is based on Jm1 = (J-1), or in the standard
// fashion.
template<typename Number, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  compute_JJm1(dealii::VectorizedArray<Number> const & Jm1)
{
  if constexpr(stable_formulation)
  {
    // J^2-1 = (J - 1) * (J - 1 + 2)
    return (Jm1 * (Jm1 + 2.0));
  }
  else
  {
    // J^2-1 = (J - 1 + 1) * (J - 1 + 1) - 1
    return ((Jm1 + 1.0) * (Jm1 + 1.0) - 1.0);
  }
}

// Compute I_1 = trace(C) in a numerically stable manner, which is based on E, or in the standard
// fashion.
template<int dim, typename Number, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  compute_I_1(dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> const & E)
{
  if constexpr(stable_formulation)
  {
    // I_1 = trace(C) = 2 * trace(E) + trace(I)
    return (2.0 * trace(E) + static_cast<Number>(dim));
  }
  else
  {
    // I_1 = trace(C) = trace(2 * E + I)
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> C = 2.0 * E;
    add_scaled_identity(C, static_cast<Number>(1.0));
    return trace(C);
  }
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  std::pair<Number, bool>
  solve_polynomial_newton(Number const &     a,
                          Number const &     b,
                          Number const &     c,
                          Number const &     d,
                          Number const &     absolute_tolerance = 1e-9,
                          Number const &     relative_tolerance = 1e-3,
                          unsigned int const max_iterations     = 10,
                          Number const &     initial_guess      = 0.0)
{
  // Find smallest real-valued positive root of a * x^3 + b * x^2 + c * x + d = 0.
  auto const f    = [&](Number x) -> Number { return (a * x * x * x + b * x * x + c * x + d); };
  auto const dfdx = [&](Number x) -> Number { return (3.0 * a * x * x + 2.0 * b * x + c); };

  Number xnp           = initial_guess;
  Number residual_init = std::abs(f(xnp));

  bool converged = false;
  if(residual_init > absolute_tolerance)
  {
    // Compute admissible quasi-Newton starting tangent.
    Number dfdx_eval_old = dfdx(xnp);
    if(dfdx_eval_old < absolute_tolerance * absolute_tolerance)
    {
      // Shift by absolute tolerance.
      dfdx_eval_old = residual_init * absolute_tolerance;
    }

    // Execute (quasi-)Newton loop.
    unsigned int n      = 0;
    Number       f_eval = f(xnp);
    do
    {
      Number dfdx_eval = dfdx(xnp);

      if(dfdx_eval > absolute_tolerance * absolute_tolerance)
      {
        // Newton step.
        xnp           = xnp - f_eval / dfdx_eval;
        dfdx_eval_old = dfdx_eval;
      }
      else
      {
        // Quasi-Newton step.
        xnp = xnp - f_eval / dfdx_eval_old;
      }

      f_eval    = f(xnp);
      converged = (std::abs(f_eval) < absolute_tolerance) or
                  (std::abs(f_eval) < residual_init * relative_tolerance);
      n++;
    } while(n < max_iterations and not converged);
  }
  else
  {
    converged = true;
  }

  return std::make_pair(xnp, converged);
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Number
  get_J_tol()
{
  return 0.001;
}

// Reconstruction of admissible deformation gradient F and Jacobian.
// check_type 0   : Do not modify.
// check_type 1   : Global quasi-Newton, update linearization vector only if the complete field is
//                  valid everywhere (see nonlinear_operator: set_solution_linearization).
// check_type 2   : Local quasi-Newton, update use old F and J if the current one is invalid.
// check_type >2  : In-place modification, see `compute_modified_F_Jm1()` below.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  void
  reconstruct_admissible_F_Jm1(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & F,
                               dealii::VectorizedArray<Number> &                         Jm1)
{
  if constexpr(check_type < 3)
  {
    AssertThrow(check_type > 2,
                dealii::ExcMessage("Reconstruction of F and J at quadrature point "
                                   "level not intended for `check_type` in [0, 1]."));
  }

  Number tol = get_J_tol<Number>();

  // TODO: vectorize the following operations whenever possible.
  for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); ++n)
  {
    if(Jm1[n] + 1.0 <= tol)
    {
      if constexpr(check_type == 3)
      {
        // check_type 3 : Only return J = tol, while F is not modified.
        Jm1[n] = tol - 1.0;
      }
      else if constexpr(check_type == 4)
      {
        // check_type 4 : Always update, but enforce J = tol by adding a scaled unit matrix.
        // Scale factor determined by solving for the positive root of the quadratic/cubic
        // polynomial via an exact formula.
        Number fac;
        if constexpr(dim == 2)
        {
          // Find positive root of x^2 + p * x + q = 0.
          // The smaller root will always be negative related to complete self-penetration of the
          // deformation state, which we are not interested in.
          Number const p = F[0][0][n] + F[1][1][n];
          Number const q = F[0][0][n] * F[1][1][n] - F[0][1][n] * F[1][0][n] - tol;
          fac            = -p * 0.5 + sqrt(p * p * 0.25 - q);
        }
        else if constexpr(dim == 3)
        {
          // Find smallest real-valued positive root of x^3 + b * x^2 + c * x + d = 0.
          Number const b = F[0][0][n] + F[1][1][n] + F[2][2][n];
          Number const c = F[0][0][n] * F[1][1][n] - F[0][1][n] * F[1][0][n] +
                           F[0][0][n] * F[2][2][n] - F[0][2][n] * F[2][0][n] +
                           F[1][1][n] * F[2][2][n] - F[1][2][n] * F[2][1][n];
          Number const d =
            F[0][0][n] * F[1][1][n] * F[2][2][n] - F[0][0][n] * F[1][2][n] * F[2][1][n] -
            F[0][1][n] * F[1][0][n] * F[2][2][n] + F[0][1][n] * F[1][2][n] * F[2][0][n] +
            F[0][2][n] * F[1][0][n] * F[2][1][n] - F[0][2][n] * F[1][1][n] * F[2][0][n] - tol;

          Number const Q = (b * b - 3.0 * c) * ONE_NINTH;
          Number const R = (2.0 * b * b * b - 9.0 * b * c + 27.0 * d) * 0.5 * ONE_NINTH * ONE_THIRD;
          Number const Qcubed = Q * Q * Q;
          Number const a4     = Qcubed - R * R;

          if(a4 > 0)
          {
            // Three real roots, return smallest positive one.
            Number const theta = std::acos(R / std::sqrt(Qcubed));
            Number const sqrtQ = std::sqrt(Q);

            std::vector<Number> tmp(3);
            tmp[0] = -2.0 * sqrtQ * std::cos(theta * ONE_THIRD) - b * ONE_THIRD;
            tmp[1] = -2.0 * sqrtQ * std::cos((theta + 2.0 * dealii::numbers::PI) * ONE_THIRD) -
                     b * ONE_THIRD;
            tmp[2] = -2.0 * sqrtQ * std::cos((theta + 4.0 * dealii::numbers::PI) * ONE_THIRD) -
                     b * ONE_THIRD;

            fac = std::numeric_limits<Number>::max();
            for(unsigned int i = 0; i < 3; ++i)
            {
              if(tmp[i] > 0 and tmp[i] < fac)
              {
                fac = tmp[i];
              }
            }
          }
          else
          {
            // Single real root.
            Number e = std::exp(ONE_THIRD * std::log(std::sqrt(-a4) + std::abs(R)));
            e        = R > 0 ? -e : e;
            fac      = (e + Q / e) - b * ONE_THIRD;
          }
        }
        else
        {
          AssertThrow(dim != 2 and dim != 3,
                      dealii::ExcMessage("Unexpected dim. Choose dim == 2 or dim == 3."));
        }

        for(unsigned int d = 0; d < dim; ++d)
        {
          F[d][d][n] += fac;
        }

        // J = tol follows by construction.
        Jm1[n] = tol - 1.0;
      }
      else if constexpr(check_type == 5)
      {
        // check_type 5 : Always update, but enforce J = tol by adding a scaled unit matrix.
        // Scale factor determined by solving for the positive root of the quadratic/cubic
        // polynomial via a Newton solver.
        Number fac;
        bool   converged;
        if constexpr(dim == 2)
        {
          // Find positive root of x^2 + p * x + q = 0.
          Number const p = F[0][0][n] + F[1][1][n];
          Number const q = F[0][0][n] * F[1][1][n] - F[0][1][n] * F[1][0][n] - tol;
          std::tie(fac, converged) =
            solve_polynomial_newton<Number>(0, 1.0, p, q, tol * tol, tol, 5, 0.0);
        }
        else if constexpr(dim == 3)
        {
          // Find smallest real-valued positive root of x^3 + b * x^2 + c * x + d = 0.
          Number const b = F[0][0][n] + F[1][1][n] + F[2][2][n];
          Number const c = F[0][0][n] * F[1][1][n] - F[0][1][n] * F[1][0][n] +
                           F[0][0][n] * F[2][2][n] - F[0][2][n] * F[2][0][n] +
                           F[1][1][n] * F[2][2][n] - F[1][2][n] * F[2][1][n];
          Number const d =
            F[0][0][n] * F[1][1][n] * F[2][2][n] - F[0][0][n] * F[1][2][n] * F[2][1][n] -
            F[0][1][n] * F[1][0][n] * F[2][2][n] + F[0][1][n] * F[1][2][n] * F[2][0][n] +
            F[0][2][n] * F[1][0][n] * F[2][1][n] - F[0][2][n] * F[1][1][n] * F[2][0][n] - tol;
          std::tie(fac, converged) =
            solve_polynomial_newton<Number>(1.0, b, c, d, tol * tol, tol, 5, 0.0);
        }
        else
        {
          AssertThrow(dim != 2 and dim != 3,
                      dealii::ExcMessage("Unexpected dim. Choose dim == 2 or dim == 3."));
        }

        if(converged)
        {
          for(unsigned int d = 0; d < dim; ++d)
          {
            F[d][d][n] += fac;
          }
        }
        else
        {
          std::cout << "Newton algorithm did not converge.\n";
        }

        // J = tol follows by construction.
        Jm1[n] = tol - 1.0;
      }
      else if(check_type == 6)
      {
        // check_type 6 : always update, but enforce J = tol by eigenvalue decomposition.
        AssertThrow(false, dealii::ExcNotImplemented());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("This check_type is not defined."));
      }
    }
  }
}

// The following functions return F and J after potential modification
// and can be seen as utility functions. Note that in order to check
// the admissibility of the deformation gradient F, we need to compute
// the Jacobian J, regardless of possible return value. Therefore,
// several versions are implemented below, where the `check_type` is
// templated, such that no additional work/initialization is done within
// these functions for check_type < 3.

// This version always returns F and Jm1 after potential modification.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  std::tuple<dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>,
             dealii::VectorizedArray<Number>>
  compute_modified_F_Jm1(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> F = compute_F(gradient_displacement);
  dealii::VectorizedArray<Number>                         Jm1 =
    compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);

  if constexpr(check_type > 2)
  {
    reconstruct_admissible_F_Jm1<dim, Number, check_type, stable_formulation>(F, Jm1);
  }

  return {F, Jm1};
}

// This version returns only F after potential modification.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  compute_modified_F(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  if constexpr(check_type < 3)
  {
    return compute_F(gradient_displacement);
  }
  else
  {
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> F = compute_F(gradient_displacement);

    dealii::VectorizedArray<Number> Jm1 =
      compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);

    reconstruct_admissible_F_Jm1<dim, Number, check_type, stable_formulation>(F, Jm1);

    return F;
  }
}

// This version returns only Jm1 = max(J - 1, tol - 1) as by construction.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  compute_modified_Jm1(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  if constexpr(check_type < 3)
  {
    return compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
  }
  else
  {
    dealii::VectorizedArray<Number> Jm1 =
      compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
    Number tol = get_J_tol<Number>();

    if constexpr(check_type < 6)
    {
      return std::max(Jm1, dealii::make_vectorized_array(static_cast<Number>(tol - 1.0)));
    }
    else
    {
      AssertThrow(check_type < 6,
                  dealii::ExcMessage("Correction of Jm1 not implemented this `check_type`."));
      return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
    }
  }
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
