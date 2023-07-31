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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_RESOLUTION_PARAMETERS_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_RESOLUTION_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
struct TemporalResolutionParameters
{
  TemporalResolutionParameters()
  {
  }

  TemporalResolutionParameters(std::string const & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    prm.enter_subsection("TemporalResolution");
    {
      prm.add_parameter("RefineTimeMin",
                        refine_time_min,
                        "Minimal number of time refinements.",
                        dealii::Patterns::Integer(0, 20),
                        true);
      prm.add_parameter("RefineTimeMax",
                        refine_time_max,
                        "Maximal number of time refinements.",
                        dealii::Patterns::Integer(0, 20),
                        true);
    }
    prm.leave_subsection();
  }

  unsigned int refine_time_min = 0;
  unsigned int refine_time_max = 0;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_TIME_INTEGRATION_RESOLUTION_PARAMETERS_H_ */
