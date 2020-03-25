/*
 * convergence_study.h
 *
 *  Created on: 25.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_CONVERGENCE_STUDY_H_
#define INCLUDE_FUNCTIONALITIES_CONVERGENCE_STUDY_H_

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

struct ConvergenceStudy
{
  ConvergenceStudy()
  {
  }

  ConvergenceStudy(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);

    parse_input(input_file, prm, true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("General");
      prm.add_parameter("Precision",      precision,        "Floating point precision.",                     Patterns::Selection("float|double"));
      prm.add_parameter("Dim",            dim,              "Number of space dimension.",                    Patterns::Integer(2,3));
      prm.add_parameter("DegreeMin",      degree_min,       "Minimal polynomial degree of shape functions.", Patterns::Integer(1,15));
      prm.add_parameter("DegreeMax",      degree_max,       "Maximal polynomial degree of shape functions.", Patterns::Integer(1,15));
      prm.add_parameter("RefineSpaceMin", refine_space_min, "Minimal number of mesh refinements.",           Patterns::Integer(0,20));
      prm.add_parameter("RefineSpaceMax", refine_space_max, "Maximal number of mesh refinements.",           Patterns::Integer(0,20));
      prm.add_parameter("RefineTimeMin",  refine_time_min,  "Minimal number of time refinements.",           Patterns::Integer(0,20));
      prm.add_parameter("RefineTimeMax",  refine_time_max,  "Maximal number of time refinements.",           Patterns::Integer(0,20));
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  unsigned int degree_min = 3;
  unsigned int degree_max = 3;

  unsigned int refine_space_min = 0;
  unsigned int refine_space_max = 0;

  unsigned int refine_time_min = 0;
  unsigned int refine_time_max = 0;
};



#endif /* INCLUDE_FUNCTIONALITIES_CONVERGENCE_STUDY_H_ */
