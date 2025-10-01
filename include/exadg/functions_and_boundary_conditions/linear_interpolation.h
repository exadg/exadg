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

#ifndef EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_LINEAR_INTERPOLATION_H_
#define EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_LINEAR_INTERPOLATION_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// C/C++
#include <vector>

namespace ExaDG
{
template<int dim, typename Number>
Number
linear_interpolation_1d(double const &                                      y,
                        std::vector<Number> const &                         y_values,
                        std::vector<dealii::Tensor<1, dim, Number>> const & solution_values,
                        unsigned int const &                                component);

/*
 *  2D interpolation for rectangular cross-sections
 */
template<int dim, typename Number>
Number
linear_interpolation_2d_cartesian(
  dealii::Point<dim> const &                          point,
  std::vector<Number> const &                         y_values,
  std::vector<Number> const &                         z_values,
  std::vector<dealii::Tensor<1, dim, Number>> const & solution_values,
  unsigned int const &                                component);

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
  unsigned int const &                                component);

} // namespace ExaDG

#endif /* EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_LINEAR_INTERPOLATION_H_ */
