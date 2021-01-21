/*
 * convergence_study.h
 *
 *  Created on: 25.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_UTILITIES_RESOLUTION_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_RESOLUTION_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
using namespace dealii;

struct SpatialResolutionParameters
{
  SpatialResolutionParameters()
  {
  }

  SpatialResolutionParameters(const std::string & input_file)
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
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("DegreeMax",
                        degree_max,
                        "Maximal polynomial degree of shape functions.",
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("RefineSpaceMin",
                        refine_space_min,
                        "Minimal number of mesh refinements.",
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineSpaceMax",
                        refine_space_max,
                        "Maximal number of mesh refinements.",
                        Patterns::Integer(0,20),
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

  TemporalResolutionParameters(const std::string & input_file)
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
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineTimeMax",
                        refine_time_max,
                        "Maximal number of time refinements.",
                        Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  unsigned int refine_time_min = 0;
  unsigned int refine_time_max = 0;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_RESOLUTION_PARAMETERS_H_ */
