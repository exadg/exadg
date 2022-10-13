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
namespace numbers
{
unsigned int const steady_timestep = -1;
}

namespace Utilities
{
inline bool
is_unsteady_timestep(unsigned int const timestep)
{
  return (timestep != numbers::steady_timestep);
}
} // namespace Utilities
} // namespace ExaDG

#endif /*INCLUDE_EXADG_UTILITIES_NUMBERS_H_*/
