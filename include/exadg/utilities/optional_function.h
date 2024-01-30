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
 * This is a @c dealii::Function, that provides information if its result is constant over space at
 * every time.
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
   * Varies in space, i.e. is not constant over space at every time.
   */
  virtual bool
  varies_in_space() const = 0;

  /**
   * Compute the value that is constant over space at every time.
   */
  virtual Number
  compute_time_factor() const = 0;
};

} // namespace Utilities
} // namespace ExaDG

#endif /*EXADG_AERO_UTILITIES_SPATIAL_AWARE_FUNCTION_H_*/
