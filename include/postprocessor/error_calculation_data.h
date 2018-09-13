/*
 * ErrorCalculationData.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_ERROR_CALCULATION_DATA_H_
#define INCLUDE_POSTPROCESSOR_ERROR_CALCULATION_DATA_H_

#include "../functionalities/print_functions.h"

struct ErrorCalculationData
{
  ErrorCalculationData()
    :
    analytical_solution_available(false),
    calculate_relative_errors(true),
    calculate_H1_seminorm_velocity(false),
    error_calc_start_time(std::numeric_limits<double>::max()),
    error_calc_interval_time(std::numeric_limits<double>::max()),
    calculate_every_time_steps(std::numeric_limits<unsigned int>::max()),
    write_errors_to_file(false),
    filename_prefix("error")
  {}

  void print(ConditionalOStream &pcout, bool unsteady)
  {
    print_parameter(pcout,"Calculate error",analytical_solution_available);
    if(analytical_solution_available == true && unsteady == true)
    {
      print_parameter(pcout,"Calculate relative errors",calculate_relative_errors);
      print_parameter(pcout,"Calculate H1-seminorm velocity",calculate_H1_seminorm_velocity);
      print_parameter(pcout,"Error calculation start time",error_calc_start_time);
      print_parameter(pcout,"Error calculation interval time",error_calc_interval_time);
      print_parameter(pcout,"Calculate error every time steps",calculate_every_time_steps);
      print_parameter(pcout,"Write errors to file",write_errors_to_file);
      print_parameter(pcout, "Filename", filename_prefix);
    }
  }

  // to calculate the error an analytical solution to the problem has to be available
  bool analytical_solution_available;

  // relative or absolute errors?
  // If calculate_relative_errors == false, this implies that absolute errors are calculated
  bool calculate_relative_errors;

  // by default, the L2-errors are computed for velocity and pressure. Other norms
  // have to be explicitly specified by the user
  bool calculate_H1_seminorm_velocity;

  // before then no error calculation will be performed
  double error_calc_start_time;

  // specifies the time interval in which error calculation is performed
  double error_calc_interval_time;

  // calculate error every time steps
  unsigned int calculate_every_time_steps;

  // write errors to file?
  bool write_errors_to_file;

  // filename
  std::string filename_prefix;
};


#endif /* INCLUDE_POSTPROCESSOR_ERROR_CALCULATION_DATA_H_ */
