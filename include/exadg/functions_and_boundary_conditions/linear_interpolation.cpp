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

#include <exadg/functions_and_boundary_conditions/linear_interpolation.h>

namespace ExaDG
{
template<int dim, typename Number>
Number
linear_interpolation_1d(double const &                                      y,
                        std::vector<Number> const &                         y_values,
                        std::vector<dealii::Tensor<1, dim, Number>> const & solution_values,
                        unsigned int const &                                component)
{
  Number result = 0.0;

  Number const tol = 1.e-2;

  unsigned int const n_points_y = y_values.size();

  AssertThrow((y_values[0] - tol < y) and (y < y_values[n_points_y - 1] + tol),
              dealii::ExcMessage("invalid point found."));

  // interpolate y-coordinates
  unsigned int iy = 0;

  iy = std::distance(y_values.begin(), std::lower_bound(y_values.begin(), y_values.end(), y));
  // make sure that the index does not exceed the array bounds in case of round-off errors
  if(iy == y_values.size())
    iy--;

  if(iy == 0)
    iy++;

  Number const weight_yp = (y - y_values[iy - 1]) / (y_values[iy] - y_values[iy - 1]);
  Number const weight_ym = 1 - weight_yp;

  AssertThrow(-1.e-12 < weight_yp and weight_yp < 1. + 1e-12 and -1.e-12 < weight_ym and
                weight_ym < 1. + 1e-12,
              dealii::ExcMessage("invalid weights when interpolating solution in 1D."));

  result =
    weight_ym * solution_values[iy - 1][component] + weight_yp * solution_values[iy][component];

  return result;
}

template float
linear_interpolation_1d(double const &                                   y,
                        std::vector<float> const &                       y_values,
                        std::vector<dealii::Tensor<1, 2, float>> const & solution_values,
                        unsigned int const &                             component);

template double
linear_interpolation_1d(double const &                                    y,
                        std::vector<double> const &                       y_values,
                        std::vector<dealii::Tensor<1, 2, double>> const & solution_values,
                        unsigned int const &                              component);

template float
linear_interpolation_1d(double const &                                   y,
                        std::vector<float> const &                       y_values,
                        std::vector<dealii::Tensor<1, 3, float>> const & solution_values,
                        unsigned int const &                             component);

template double
linear_interpolation_1d(double const &                                    y,
                        std::vector<double> const &                       y_values,
                        std::vector<dealii::Tensor<1, 3, double>> const & solution_values,
                        unsigned int const &                              component);

template<int dim, typename Number>
Number
linear_interpolation_2d_cartesian(
  dealii::Point<dim> const &                          point,
  std::vector<Number> const &                         y_values,
  std::vector<Number> const &                         z_values,
  std::vector<dealii::Tensor<1, dim, Number>> const & solution_values,
  unsigned int const &                                component)
{
  AssertThrow(dim == 3, dealii::ExcMessage("not implemented"));

  Number result = 0.0;

  Number const tol = 1.e-10;

  unsigned int const n_points_y = y_values.size();
  unsigned int const n_points_z = z_values.size();

  // make sure that point does not exceed bounds
  dealii::Point<dim> p = point;

  AssertThrow((y_values[0] - tol < p[1]) and (p[1] < y_values[n_points_y - 1] + tol) and
                (z_values[0] - tol < p[2]) and (p[2] < z_values[n_points_z - 1] + tol),
              dealii::ExcMessage("invalid point found."));

  // interpolate y and z-coordinates
  unsigned int iy = 0, iz = 0;

  iy = std::distance(y_values.begin(), std::lower_bound(y_values.begin(), y_values.end(), p[1]));
  iz = std::distance(z_values.begin(), std::lower_bound(z_values.begin(), z_values.end(), p[2]));
  // make sure that the index does not exceed the array bounds in case of round-off errors
  if(iy == y_values.size())
    iy--;
  if(iz == z_values.size())
    iz--;

  if(iy == 0)
    iy++;
  if(iz == 0)
    iz++;

  Number const weight_yp = (p[1] - y_values[iy - 1]) / (y_values[iy] - y_values[iy - 1]);
  Number const weight_ym = 1 - weight_yp;
  Number const weight_zp = (p[2] - z_values[iz - 1]) / (z_values[iz] - z_values[iz - 1]);
  Number const weight_zm = 1 - weight_zp;

  AssertThrow(-1.e-12 < weight_yp and weight_yp < 1. + 1e-12 and -1.e-12 < weight_ym and
                weight_ym < 1. + 1e-12 and -1.e-12 < weight_zp and weight_zp < 1. + 1e-12 and
                -1.e-12 < weight_zm and weight_zm < 1. + 1e-12,
              dealii::ExcMessage("invalid weights when interpolating solution in 2D."));

  result = weight_ym * weight_zm * solution_values[(iy - 1) * n_points_z + (iz - 1)][component] +
           weight_ym * weight_zp * solution_values[(iy - 1) * n_points_z + (iz)][component] +
           weight_yp * weight_zm * solution_values[(iy)*n_points_z + (iz - 1)][component] +
           weight_yp * weight_zp * solution_values[(iy)*n_points_z + (iz)][component];

  return result;
}

template float
linear_interpolation_2d_cartesian(dealii::Point<2> const &                         point,
                                  std::vector<float> const &                       y_values,
                                  std::vector<float> const &                       z_values,
                                  std::vector<dealii::Tensor<1, 2, float>> const & solution_values,
                                  unsigned int const &                             component);

template double
linear_interpolation_2d_cartesian(dealii::Point<2> const &                          point,
                                  std::vector<double> const &                       y_values,
                                  std::vector<double> const &                       z_values,
                                  std::vector<dealii::Tensor<1, 2, double>> const & solution_values,
                                  unsigned int const &                              component);

template float
linear_interpolation_2d_cartesian(dealii::Point<3> const &                         point,
                                  std::vector<float> const &                       y_values,
                                  std::vector<float> const &                       z_values,
                                  std::vector<dealii::Tensor<1, 3, float>> const & solution_values,
                                  unsigned int const &                             component);

template double
linear_interpolation_2d_cartesian(dealii::Point<3> const &                          point,
                                  std::vector<double> const &                       y_values,
                                  std::vector<double> const &                       z_values,
                                  std::vector<dealii::Tensor<1, 3, double>> const & solution_values,
                                  unsigned int const &                              component);

/*
 *  2D interpolation for cylindrical cross-sections
 */
template<int dim, typename Number>
Number
linear_interpolation_2d_cylindrical(
  Number const                                        r_in,
  Number const                                        phi,
  std::vector<Number> const &                         r_values,
  std::vector<Number> const &                         phi_values,
  std::vector<dealii::Tensor<1, dim, Number>> const & solution_values,
  unsigned int const &                                component)
{
  AssertThrow(dim == 3, dealii::ExcMessage("not implemented"));

  Number result = 0.0;

  Number const tol = 1.e-10;

  unsigned int const n_points_r   = r_values.size();
  unsigned int const n_points_phi = phi_values.size();

  Number r = r_in;

  if(r > r_values[n_points_r - 1])
    r = r_values[n_points_r - 1];

  AssertThrow(r > (r_values[0] - tol) and r < (r_values[n_points_r - 1] + tol) and
                phi > (phi_values[0] - tol) and phi < (phi_values[n_points_phi - 1] + tol),
              dealii::ExcMessage("invalid point found."));

  // interpolate r and phi-coordinates
  unsigned int i_r = 0, i_phi = 0;

  i_r = std::distance(r_values.begin(), std::lower_bound(r_values.begin(), r_values.end(), r));
  i_phi =
    std::distance(phi_values.begin(), std::lower_bound(phi_values.begin(), phi_values.end(), phi));

  // make sure that the index does not exceed the array bounds in case of round-off errors
  if(i_r == r_values.size())
    i_r--;
  if(i_phi == phi_values.size())
    i_phi--;

  if(i_r == 0)
    i_r++;
  if(i_phi == 0)
    i_phi++;

  AssertThrow(i_r > 0 and i_r < n_points_r and i_phi > 0 and i_phi < n_points_phi,
              dealii::ExcMessage("Invalid point found"));

  Number const weight_r_p = (r - r_values[i_r - 1]) / (r_values[i_r] - r_values[i_r - 1]);
  Number const weight_r_m = 1 - weight_r_p;
  Number const weight_phi_p =
    (phi - phi_values[i_phi - 1]) / (phi_values[i_phi] - phi_values[i_phi - 1]);
  Number const weight_phi_m = 1 - weight_phi_p;

  AssertThrow(-1.e-12 < weight_r_p and weight_r_p < 1. + 1e-12 and -1.e-12 < weight_r_m and
                weight_r_m < 1. + 1e-12 and -1.e-12 < weight_phi_p and weight_phi_p < 1. + 1e-12 and
                -1.e-12 < weight_phi_m and weight_phi_m < 1. + 1e-12,
              dealii::ExcMessage("invalid weights when interpolating solution in 2D."));

  result =
    weight_r_m * weight_phi_m * solution_values[(i_r - 1) * n_points_phi + (i_phi - 1)][component] +
    weight_r_m * weight_phi_p * solution_values[(i_r - 1) * n_points_phi + (i_phi)][component] +
    weight_r_p * weight_phi_m * solution_values[(i_r)*n_points_phi + (i_phi - 1)][component] +
    weight_r_p * weight_phi_p * solution_values[(i_r)*n_points_phi + (i_phi)][component];

  return result;
}

template float
linear_interpolation_2d_cylindrical(
  float const                                      r_in,
  float const                                      phi,
  std::vector<float> const &                       r_values,
  std::vector<float> const &                       phi_values,
  std::vector<dealii::Tensor<1, 2, float>> const & solution_values,
  unsigned int const &                             component);

template double
linear_interpolation_2d_cylindrical(
  double const                                      r_in,
  double const                                      phi,
  std::vector<double> const &                       r_values,
  std::vector<double> const &                       phi_values,
  std::vector<dealii::Tensor<1, 2, double>> const & solution_values,
  unsigned int const &                              component);

template float
linear_interpolation_2d_cylindrical(
  float const                                      r_in,
  float const                                      phi,
  std::vector<float> const &                       r_values,
  std::vector<float> const &                       phi_values,
  std::vector<dealii::Tensor<1, 3, float>> const & solution_values,
  unsigned int const &                             component);

template double
linear_interpolation_2d_cylindrical(
  double const                                      r_in,
  double const                                      phi,
  std::vector<double> const &                       r_values,
  std::vector<double> const &                       phi_values,
  std::vector<dealii::Tensor<1, 3, double>> const & solution_values,
  unsigned int const &                              component);

} // namespace ExaDG
