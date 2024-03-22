/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2024 by the ExaDG authors
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

#ifndef EXADG_AERO_UTILITIES_SPATIAL_AWARE_FUNCTION_H_
#define EXADG_AERO_UTILITIES_SPATIAL_AWARE_FUNCTION_H_

#include <deal.II/base/function.h>

namespace ExaDG
{
namespace Utilities
{
/**
 * The intention behind is to bypass a potentially expensive evaluation of
 * @c dealii::Function::value() (which the internal implementation will call for each quadrature
 * point) in case one knows a priori that the function does not vary in space.
 */
template<int dim, typename Number = double>
class SpatialAwareFunction : public dealii::Function<dim, Number>
{
public:
  using time_type = typename dealii::Function<dim, Number>::time_type;

  SpatialAwareFunction(const unsigned int n_components = 1, const time_type initial_time = 0.0)
    : dealii::Function<dim, Number>(n_components, initial_time)
  {
  }

  /**
   * If this function evaluates to true, @c dealii::Function::set_time() in combination with
   * @c dealii::Function::value() will be used to evaluate the function's value."
   */
  virtual bool
  varies_in_space(double const time) const = 0;

  /**
   * If the above function @c varies_in_space() returns false, this function will be used to
   * evaluate the function's value, bypassing @c dealii::Function::value().
   */
  virtual Number
  compute_time_factor(double const time) const = 0;
};

} // namespace Utilities
} // namespace ExaDG

#endif /*EXADG_AERO_UTILITIES_SPATIAL_AWARE_FUNCTION_H_*/
