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

#ifndef INCLUDE_EXADG_UTILITIES_RESOLUTION_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_RESOLUTION_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/configuration/config.h>

namespace ExaDG
{
struct SpatialResolutionParameters
{
  SpatialResolutionParameters()
  {
  }

  SpatialResolutionParameters(std::string const & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("SpatialResolution");
      prm.add_parameter("DegreeMin",
                        degree_min,
                        "Minimal polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("DegreeMax",
                        degree_max,
                        "Maximal polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("RefineSpaceMin",
                        refine_space_min,
                        "Minimal number of mesh refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineSpaceMax",
                        refine_space_max,
                        "Maximal number of mesh refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  unsigned int degree_min = 3;
  unsigned int degree_max = 3;

  unsigned int refine_space_min = 0;
  unsigned int refine_space_max = 0;
};

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
    // clang-format off
    prm.enter_subsection("TemporalResolution");
      prm.add_parameter("RefineTimeMin",
                        refine_time_min,
                        "Minimal number of time refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineTimeMax",
                        refine_time_max,
                        "Maximal number of time refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  unsigned int refine_time_min = 0;
  unsigned int refine_time_max = 0;
};

struct ResolutionParameters
{
  ResolutionParameters()
  {
  }

  ResolutionParameters(std::string const & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm,
                 std::string const &        subsection_name = "SpatialResolution")
  {
    // clang-format off
    prm.enter_subsection(subsection_name);
      prm.add_parameter("Degree",
                        degree,
                        "Polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("RefineSpace",
                        refine_space,
                        "Number of global, uniform mesh refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  unsigned int degree = 3;

  unsigned int refine_space = 0;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_RESOLUTION_PARAMETERS_H_ */
