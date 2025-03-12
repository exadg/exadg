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

#ifndef INCLUDE_EXADG_UTILITIES_GENERAL_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_GENERAL_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
struct GeneralParameters
{
  GeneralParameters()
  {
  }

  GeneralParameters(std::string const & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    prm.enter_subsection("General");
    {
      prm.add_parameter("Precision",
                        precision,
                        "Floating point precision.",
                        dealii::Patterns::Selection("float|double"),
                        false);
      prm.add_parameter(
        "Dim", dim, "Number of space dimension.", dealii::Patterns::Integer(2, 3), true);
      prm.add_parameter("IsTest",
                        is_test,
                        "Set to true if the program is run as a test.",
                        dealii::Patterns::Bool(),
                        false);
    }
    prm.leave_subsection();
  }

  std::string precision = "double";

  unsigned int dim = 2;

  bool is_test = false;
};

} // namespace ExaDG



#endif /* INCLUDE_EXADG_UTILITIES_GENERAL_PARAMETERS_H_ */
