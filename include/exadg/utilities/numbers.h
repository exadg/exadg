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

#ifndef INCLUDE_EXADG_UTILITIES_NUMBERS_H_
#define INCLUDE_EXADG_UTILITIES_NUMBERS_H_

namespace ExaDG
{
namespace types
{
using time_step = unsigned int;
}

namespace numbers
{
types::time_step const invalid_timestep = std::numeric_limits<unsigned int>::max();
types::time_step const steady_timestep  = std::numeric_limits<unsigned int>::max() - 1;
} // namespace numbers

namespace Utilities
{
inline bool
is_unsteady_timestep(types::time_step const timestep)
{
  return (timestep != numbers::steady_timestep);
}
inline bool
is_valid_timestep(types::time_step const timestep)
{
  return (timestep != numbers::invalid_timestep);
}
} // namespace Utilities
} // namespace ExaDG

#endif /*INCLUDE_EXADG_UTILITIES_NUMBERS_H_*/
