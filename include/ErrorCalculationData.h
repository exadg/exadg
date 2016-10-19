/*
 * ErrorCalculationData.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_ERRORCALCULATIONDATA_H_
#define INCLUDE_ERRORCALCULATIONDATA_H_

#include "../include/PrintFunctions.h"

struct ErrorCalculationData
{
  ErrorCalculationData()
    :
    analytical_solution_available(false),
    error_calc_start_time(std::numeric_limits<double>::max()),
    error_calc_interval_time(std::numeric_limits<double>::max())
  {}

  void print(ConditionalOStream &pcout, bool unsteady)
  {
    print_parameter(pcout,"Calculate error",analytical_solution_available);
    if(analytical_solution_available == true && unsteady == true)
    {
      print_parameter(pcout,"Error calculation start time",error_calc_start_time);
      print_parameter(pcout,"Error calculation interval time",error_calc_interval_time);
    }
  }

  // to calculate the error an analytical solution to the problem has to be available
  bool analytical_solution_available;

  // before then no error calculation will be performed
  double error_calc_start_time;

  // specifies the time interval in which error calculation is performed
  double error_calc_interval_time;
};


#endif /* INCLUDE_ERRORCALCULATIONDATA_H_ */
