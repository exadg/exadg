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

#ifndef EXADG_AERO_UTILITIES_OPTIONAL_FUNCTION_H_
#define EXADG_AERO_UTILITIES_OPTIONAL_FUNCTION_H_

#include <deal.II/base/function.h>

namespace ExaDG
{
namespace Utilities
{
/**
 * This is a @c dealii::Function, that provides information if it has
 * to be evaluated at a given time.
 */

template<int dim, typename Number = double>
class OptionalFunction : public dealii::Function<dim, Number>
{
public:
  using time_type = typename dealii::Function<dim, Number>::time_type;

  OptionalFunction(const unsigned int n_components = 1, const time_type initial_time = 0.0)
    : dealii::Function<dim, Number>(n_components, initial_time)
  {
  }

  virtual bool
  needs_evaluation_at_time(double const new_time) const = 0;
};

} // namespace Utilities
} // namespace ExaDG

#endif /*EXADG_AERO_ACOUSTIC_USER_INTERFACE_BLEND_IN_FUNCTION_H_*/
