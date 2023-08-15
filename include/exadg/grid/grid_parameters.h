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

#ifndef EXADG_GRID_GRID_PARAMETERS_H
#define EXADG_GRID_GRID_PARAMETERS_H

#include <string>

#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
struct GridParameters
{
  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "Grid")
  {
    prm.enter_subsection(subsection_name);
    {
      prm.add_parameter("FileName", file_name, "External input grid file.");
    }
    prm.leave_subsection();
  }

  std::string file_name;
};

} // namespace ExaDG

#endif // EXADG_GRID_GRID_PARAMETERS_H
