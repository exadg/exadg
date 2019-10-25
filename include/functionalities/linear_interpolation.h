/*
 * linear_interpolation.h
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

template<int dim, typename Number>
Number
linear_interpolation_1d(double const &                              y,
                        std::vector<Number> const &                 y_values,
                        std::vector<Tensor<1, dim, Number>> const & solution_values,
                        unsigned int const &                        component)
{
  Number result = 0.0;

  Number const tol = 1.e-2;

  unsigned int const n_points_y = y_values.size();

  AssertThrow((y_values[0] - tol < y) && (y < y_values[n_points_y - 1] + tol),
              ExcMessage("invalid point found."));

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

  AssertThrow(-1.e-12 < weight_yp && weight_yp < 1. + 1e-12 && -1.e-12 < weight_ym &&
                weight_ym < 1. + 1e-12,
              ExcMessage("invalid weights when interpolating solution in 1D."));

  result =
    weight_ym * solution_values[iy - 1][component] + weight_yp * solution_values[iy][component];

  return result;
}

template<int dim, typename Number>
Number
linear_interpolation_2d_cartesian(Point<dim> const &                          p,
                                  std::vector<Number> const &                 y_values,
                                  std::vector<Number> const &                 z_values,
                                  std::vector<Tensor<1, dim, Number>> const & solution_values,
                                  unsigned int const &                        component)
{
  AssertThrow(dim == 3, ExcMessage("not implemented"));

  Number result = 0.0;

  Number const tol = 1.e-2;

  unsigned int const n_points_y = y_values.size();
  unsigned int const n_points_z = z_values.size();

  AssertThrow((y_values[0] - tol < p[1]) && (p[1] < y_values[n_points_y - 1] + tol) &&
                (z_values[0] - tol < p[2]) && (p[2] < z_values[n_points_z - 1] + tol),
              ExcMessage("invalid point found."));

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

  AssertThrow(-1.e-12 < weight_yp && weight_yp < 1. + 1e-12 && -1.e-12 < weight_ym &&
                weight_ym < 1. + 1e-12 && -1.e-12 < weight_zp && weight_zp < 1. + 1e-12 &&
                -1.e-12 < weight_zm && weight_zm < 1. + 1e-12,
              ExcMessage("invalid weights when interpolating solution in 2D."));

  result = weight_ym * weight_zm * solution_values[(iy - 1) * n_points_z + (iz - 1)][component] +
           weight_ym * weight_zp * solution_values[(iy - 1) * n_points_z + (iz)][component] +
           weight_yp * weight_zm * solution_values[(iy)*n_points_z + (iz - 1)][component] +
           weight_yp * weight_zp * solution_values[(iy)*n_points_z + (iz)][component];

  return result;
}

/*
 *  2D interpolation for cylindrical cross-sections
 */
template<int dim, typename Number>
Number
linear_interpolation_2d_cylindrical(Number const                                r_in,
                                    Number const                                phi,
                                    std::vector<Number> const &                 r_values,
                                    std::vector<Number> const &                 phi_values,
                                    std::vector<Tensor<1, dim, Number>> const & solution_values,
                                    unsigned int const &                        component)
{
  AssertThrow(dim == 3, ExcMessage("not implemented"));

  Number result = 0.0;

  Number const tol = 1.e-10;

  unsigned int const n_points_r   = r_values.size();
  unsigned int const n_points_phi = phi_values.size();

  Number r = r_in;

  if(r > r_values[n_points_r - 1])
    r = r_values[n_points_r - 1];

  AssertThrow(r > (r_values[0] - tol) && r < (r_values[n_points_r - 1] + tol) &&
                phi > (phi_values[0] - tol) && phi < (phi_values[n_points_phi - 1] + tol),
              ExcMessage("invalid point found."));

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

  AssertThrow(i_r > 0 && i_r < n_points_r && i_phi > 0 && i_phi < n_points_phi,
              ExcMessage("Invalid point found"));

  Number const weight_r_p = (r - r_values[i_r - 1]) / (r_values[i_r] - r_values[i_r - 1]);
  Number const weight_r_m = 1 - weight_r_p;
  Number const weight_phi_p =
    (phi - phi_values[i_phi - 1]) / (phi_values[i_phi] - phi_values[i_phi - 1]);
  Number const weight_phi_m = 1 - weight_phi_p;

  AssertThrow(-1.e-12 < weight_r_p && weight_r_p < 1. + 1e-12 && -1.e-12 < weight_r_m &&
                weight_r_m < 1. + 1e-12 && -1.e-12 < weight_phi_p && weight_phi_p < 1. + 1e-12 &&
                -1.e-12 < weight_phi_m && weight_phi_m < 1. + 1e-12,
              ExcMessage("invalid weights when interpolating solution in 2D."));

  result =
    weight_r_m * weight_phi_m * solution_values[(i_r - 1) * n_points_phi + (i_phi - 1)][component] +
    weight_r_m * weight_phi_p * solution_values[(i_r - 1) * n_points_phi + (i_phi)][component] +
    weight_r_p * weight_phi_m * solution_values[(i_r)*n_points_phi + (i_phi - 1)][component] +
    weight_r_p * weight_phi_p * solution_values[(i_r)*n_points_phi + (i_phi)][component];

  return result;
}
