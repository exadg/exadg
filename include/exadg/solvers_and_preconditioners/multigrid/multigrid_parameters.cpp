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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>

namespace ExaDG
{
bool
MultigridData::involves_h_transfer() const
{
  if(type != MultigridType::pMG && type != MultigridType::cpMG && type != MultigridType::pcMG)
    return true;
  else
    return false;
}

bool
MultigridData::involves_c_transfer() const
{
  if(type == MultigridType::hMG || type == MultigridType::pMG || type == MultigridType::hpMG ||
     type == MultigridType::phMG)
    return false;
  else
    return true;
}

bool
MultigridData::involves_p_transfer() const
{
  if(type != MultigridType::hMG && type != MultigridType::hcMG && type != MultigridType::chMG)
    return true;
  else
    return false;
}

} // namespace ExaDG
