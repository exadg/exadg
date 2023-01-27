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

#ifndef INCLUDE_EXADG_UTILITIES_INPUT_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
struct InputParameters
{
  InputParameters() : input_grid_file(), read_external_grid(false)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "Input")
  {
    // clang-format off
    prm.enter_subsection(subsection_name);
      prm.add_parameter("GridFile", input_grid_file,    "External input grid file.");
      prm.add_parameter("ReadGrid", read_external_grid, "Decides whether a grid file is read.");
    prm.leave_subsection();
    // clang-format on
  };

  std::string input_grid_file;

  bool read_external_grid;
};
} // namespace ExaDG

#endif /* INCLUDE_EXADG_UTILITIES_INPUT_PARAMETERS_H_ */
