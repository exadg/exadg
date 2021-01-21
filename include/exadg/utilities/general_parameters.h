/*
 * general_parameters.h
 *
 *  Created on: Jan 20, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_UTILITIES_GENERAL_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_GENERAL_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
using namespace dealii;

struct GeneralParameters
{
  GeneralParameters()
  {
  }

  GeneralParameters(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("General");
      prm.add_parameter("Precision",
                        precision,
                        "Floating point precision.",
                        Patterns::Selection("float|double"),
                        false);
      prm.add_parameter("Dim",
                        dim,
                        "Number of space dimension.",
                        Patterns::Integer(2,3),
                        true);
      prm.add_parameter("IsTest",
                        is_test,
                        "Type of throughput study.",
                        Patterns::Bool(),
                        false);
    prm.leave_subsection();
    // clang-format on
  }

  std::string precision = "double";

  unsigned int dim = 2;

  bool is_test = false;
};

} // namespace ExaDG



#endif /* INCLUDE_EXADG_UTILITIES_GENERAL_PARAMETERS_H_ */
