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
#define ONE_THIRD  0.33333333333333333333
#define TWO_THIRDS 0.33333333333333333333
#define ONE_NINTH  0.11111111111111111111
#define TWO_NINTHS 0.22222222222222222222
// clang-format on

// deal.II
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/physics/transformations.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  void
  bound_tensor(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & tensor,
               Number const &                                            upper_bound)
{
  for(unsigned int i = 0; i < dim; ++i)
  {
    for(unsigned int j = 0; j < dim; ++j)
    {
      tensor[i][j] = std::min(tensor[i][j], dealii::make_vectorized_array(upper_bound));
    }
  }
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  log1p(dealii::VectorizedArray<Number> const & x)
{
  Number values[dealii::VectorizedArray<Number>::size()];
  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
  {
    values[i] = std::log1p(x[i]);
  }

  dealii::VectorizedArray<Number> out;
  out.load(&values[0]);
  return out;
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  expm1_limited(dealii::VectorizedArray<Number> const & x, Number const & upper_bound)
{
  Number values[dealii::VectorizedArray<Number>::size()];
  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
  {
    values[i] = std::expm1(x[i]);
    values[i] = std::min(values[i], upper_bound);
  }

  dealii::VectorizedArray<Number> out;
  out.load(&values[0]);
  return out;
}

template<typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  exp_limited(dealii::VectorizedArray<Number> const & x, Number const & upper_bound)
{
  Number values[dealii::VectorizedArray<Number>::size()];
  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
  {
    values[i] = std::exp(x[i]);
    values[i] = std::min(values[i], upper_bound);
  }

  dealii::VectorizedArray<Number> out;
  out.load(&values[0]);
  return out;
}

template<int dim, typename Number, typename TypeScale>
inline DEAL_II_ALWAYS_INLINE //
  void
  add_scaled_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & tmp,
                      TypeScale const &                                         scale)
{
  for(unsigned int i = 0; i < dim; i++)
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
  add_scaled_identity<dim, Number, Number>(F, 1.0);
  return F;
}

template<int dim, typename Number, typename TypeScale, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_E_scaled(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    TypeScale const &                                               scale)
{
  if constexpr(stable_formulation)
  {
    // E = 0.5 * (H + H^T + H^T * H)
    // where H = gradient_displacement
    return ((0.5 * scale) * (gradient_displacement + transpose(gradient_displacement) +
                             transpose(gradient_displacement) * gradient_displacement));
  }
  else
  {
    // E = 0.5 * (F^T * F - I)
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> E = compute_F(gradient_displacement);

    E = transpose(E) * E;
    add_scaled_identity<dim, Number, Number>(E, -1.0);
    return (E * (0.5 * scale));
  }
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_identity_tensor()
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> identity_tensor;
  add_scaled_identity<dim, Number, Number>(identity_tensor, 1.0);
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
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::VectorizedArray<Number>
  get_I_1(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & E,
          bool const                                                      stable_formulation)
{
  if(stable_formulation)
  {
    // I_1 = trace(C) = 2 * trace(E) + trace(I)
    return (2.0 * trace(E) + static_cast<Number>(dim));
  }
  else
  {
    // I_1 = trace(C) = trace(2 * E + I)
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> C = 2.0 * E;
    add_scaled_identity<dim, Number, Number>(C, 1.0);
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
// check_type > 1 : In-place modification, see `compute_modified_F_Jm1()` below.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  void
  reconstruct_admissible_F_Jm1(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & F,
                               dealii::VectorizedArray<Number> &                         Jm1)
{
  if constexpr(check_type < 2)
  {
    AssertThrow(check_type > 1,
                dealii::ExcMessage("Reconstruction of F and J at quadrature point "
                                   "level not intended for `check_type` in [0, 1]."));
  }

  Number tol = get_J_tol<Number>();

  // TODO: vectorize the following operations whenever possible.
  for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); ++n)
  {
    if(Jm1[n] + 1.0 <= tol)
    {
      if constexpr(check_type == 2)
      {
        // check_type 2 : Only return J = tol, while F is not modified.
        Jm1[n] = tol - 1.0;
      }
      else if constexpr(check_type == 3)
      {
        // check_type 3 : Always update, but enforce J = tol by adding a scaled unit matrix.
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
      else if constexpr(check_type == 4)
      {
        // check_type 4 : Always update, but enforce J = tol by adding a scaled unit matrix.
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
      else if(check_type == 5)
      {
        // check_type 5 : always update, but enforce J = tol by eigenvalue decomposition.
        AssertThrow(false, dealii::ExcNotImplemented());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("This check_type is not defined."));
      }
    }
  }
}

// This version always F and potentially Jm1 (depending on `compute_J`) after potential
// modification.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  void
  compute_modified_F_Jm1(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> &       F,
    dealii::VectorizedArray<Number> &                               Jm1,
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    bool const                                                      compute_J)
{
  F = compute_F(gradient_displacement);

  if(compute_J)
  {
    Jm1 = compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
  }

  if constexpr(check_type > 1)
  {
    if(not compute_J)
    {
      // Compute J - 1 to do any checking.
      Jm1 = compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
    }
    reconstruct_admissible_F_Jm1(F, Jm1);
  }
}

// This version always F and Jm1 after potential modification.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation, bool compute_J>
inline DEAL_II_ALWAYS_INLINE //
  void
  compute_modified_F_Jm1(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> &       F,
    dealii::VectorizedArray<Number> &                               Jm1,
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  F = compute_F(gradient_displacement);

  if constexpr(compute_J)
  {
    Jm1 = compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
  }

  if constexpr(check_type > 1)
  {
    if constexpr(not compute_J)
    {
      // Compute J - 1 to do any checking.
      Jm1 = compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
    }
    reconstruct_admissible_F_Jm1(F, Jm1);
  }
}

// This version returns only F after potential modification.
template<int dim, typename Number, unsigned int check_type, bool stable_formulation>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  compute_modified_F(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  if constexpr(check_type < 2)
  {
    return compute_F(gradient_displacement);
  }
  else
  {
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> F = compute_F(gradient_displacement);

    // Compute J - 1 to do any checking.
    dealii::VectorizedArray<Number> Jm1 =
      compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);

    reconstruct_admissible_F_Jm1(F, Jm1);

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
  if constexpr(check_type < 2)
  {
    return compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
  }
  else
  {
    dealii::VectorizedArray<Number> Jm1 =
      compute_Jm1<dim, Number, stable_formulation>(gradient_displacement);
    Number tol = get_J_tol<Number>();

    if constexpr(check_type < 5)
    {
      return std::max(Jm1, dealii::make_vectorized_array(static_cast<Number>(tol - 1.0)));
    }
    else
    {
      AssertThrow(check_type < 5,
                  dealii::ExcMessage("Correction of Jm1 not implemented this `check_type`."));
      return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
    }
  }
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
