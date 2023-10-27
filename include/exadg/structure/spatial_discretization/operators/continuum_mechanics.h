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

// deal.II
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/physics/transformations.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  void
  add_scaled_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> & tmp,
                      dealii::VectorizedArray<Number> const &                   factor)
{
  for(unsigned int i = 0; i < dim; i++)
  {
    tmp[i][i] = tmp[i][i] + factor;
  }
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  add_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & tmp)
{
  auto result = tmp;
  for(unsigned int i = 0; i < dim; i++)
  {
    result[i][i] = result[i][i] + 1.0;
  }
  return result;
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  subtract_identity(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & tmp)
{
  auto result = tmp;
  for(unsigned int i = 0; i < dim; i++)
  {
    result[i][i] = result[i][i] - 1.0;
  }
  return result;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_F(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement)
{
  return add_identity(gradient_displacement);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_E(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & F)
{
  return (0.5 * subtract_identity(transpose(F) * F));
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  get_identity()
{
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> I;
  return add_identity(I);
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
inline
Number
get_J_tol()
{
  return 0.001;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  void
  get_modified_F_J(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> &       F,
    dealii::VectorizedArray<Number> &                               J,
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    unsigned int const                                              check_type,
    bool const                                                      compute_J)
{
  F = add_identity(gradient_displacement);

  if(compute_J)
  {
    J = determinant(F);
  }

  // check_type 0 : Do not modify.

  // check_type 1 : Global quasi-Newton, update linearization vector only if the complete field is
  // valid everywhere (see nonlinear_operator: set_solution_linearization).

  // check_type 2 : Just update F and J values, if J > 0 (see do_set_cell_linearization_data),
  // otherwise keep the old values which are initialized with a zero displacement field at
  // simulation start.

  if(compute_J and check_type > 2)
  {
    Number tol = get_J_tol<Number>();

    for(unsigned int n = 0; n < dealii::VectorizedArray<Number>::size(); ++n)
    {
      if(J[n] <= tol)
      {
        if(check_type == 3)
        {
          // check_type 3 : Only return J = tol, while F is not modified.
          J[n] = tol;
        }
        else if(check_type == 4)
        {
          // check_type 4 : Always update, but enforce J = tol by adding a scaled unit matrix.
          // Scale factor determined by solving for the positive root of the quadratic/cubic
          // polynomial via an exact formula.
          Number fac;
          if(dim == 2)
          {
            // Find positive root of x^2 + p * x + q = 0.
            // The smaller root will always be negative related to complete self-penetration of the
            // deformation state, which we are not interested in.
            Number const p = F[0][0][n] + F[1][1][n];
            Number const q = F[0][0][n] * F[1][1][n] - F[0][1][n] * F[1][0][n] - tol;
            fac            = -p * 0.5 + sqrt(p * p * 0.25 - q);
          }
          else // dim == 3
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

            Number const one_third = 1.0 / 3.0;
            Number const Q         = (b * b - 3.0 * c) * one_third * one_third;
            Number const R =
              (2.0 * b * b * b - 9.0 * b * c + 27.0 * d) * 0.5 * one_third * one_third * one_third;
            Number const Qcubed = Q * Q * Q;
            Number const a4     = Qcubed - R * R;

            if(a4 > 0)
            {
              // Three real roots, return smallest positive one.
              Number const theta = std::acos(R / std::sqrt(Qcubed));
              Number const sqrtQ = std::sqrt(Q);

              std::vector<Number> tmp(3);
              tmp[0] = -2.0 * sqrtQ * std::cos(theta * one_third) - b * one_third;
              tmp[1] = -2.0 * sqrtQ * std::cos((theta + 2.0 * dealii::numbers::PI) * one_third) -
                       b * one_third;
              tmp[2] = -2.0 * sqrtQ * std::cos((theta + 4.0 * dealii::numbers::PI) * one_third) -
                       b * one_third;

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
              Number e = std::exp(one_third * std::log(std::sqrt(-a4) + std::abs(R)));
              e        = R > 0 ? -e : e;
              fac      = (e + Q / e) - b * one_third;
            }
          }

          for(unsigned int d = 0; d < dim; ++d)
          {
            F[d][d][n] += fac;
          }

          // J = tol follows by construction.
          J[n] = tol;

          // Alternatively, update Jacobian after modification.
          if(false)
          {
            dealii::Tensor<2, dim, Number> F_mod;
            for(unsigned int i = 0; i < dim; ++i)
            {
              for(unsigned int j = 0; j < dim; ++j)
              {
                F_mod[i][j] = F[i][j][n];
              }
            }
            J[n] = determinant(F_mod);
          }
        }
        else if(check_type == 5)
        {
          // check_type 5 : Always update, but enforce J = tol by adding a scaled unit matrix.
          // Scale factor determined by solving for the positive root of the quadratic/cubic
          // polynomial via a Newton solver.
          Number fac;
          bool   converged;
          if(dim == 2)
          {
            // Find positive root of x^2 + p * x + q = 0.
            Number const p = F[0][0][n] + F[1][1][n];
            Number const q = F[0][0][n] * F[1][1][n] - F[0][1][n] * F[1][0][n] - tol;
            std::tie(fac, converged) =
              solve_polynomial_newton<Number>(0, 1.0, p, q, tol * tol, tol, 5, 0.0);
          }
          else // dim == 3
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
          J[n] = tol;

          // Alternatively, update Jacobian after modification.
          if(false)
          {
            dealii::Tensor<2, dim, Number> F_mod;
            for(unsigned int i = 0; i < dim; ++i)
            {
              for(unsigned int j = 0; j < dim; ++j)
              {
                F_mod[i][j] = F[i][j][n];
              }
            }
            J[n] = determinant(F_mod);
          }
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
}

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
