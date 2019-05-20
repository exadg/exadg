/*
 * OutputData.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_OUTPUT_DATA_H_
#define INCLUDE_POSTPROCESSOR_OUTPUT_DATA_H_

#include "../functionalities/print_functions.h"

using namespace dealii;

struct OutputDataBase
{
  OutputDataBase()
    : write_output(false),
      output_counter_start(0),
      output_folder("output"),
      output_name("name"),
      output_start_time(std::numeric_limits<double>::max()),
      output_interval_time(std::numeric_limits<double>::max()),
      write_higher_order(true),
      degree(1)
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    // output for visualization of results
    print_parameter(pcout, "Write output", write_output);

    if(write_output == true)
    {
      print_parameter(pcout, "Output counter start", output_counter_start);
      print_parameter(pcout, "Output folder", output_folder);
      print_parameter(pcout, "Name of output files", output_name);

      if(unsteady == true)
      {
        print_parameter(pcout, "Output start time", output_start_time);
        print_parameter(pcout, "Output interval time", output_interval_time);
      }

      print_parameter(pcout, "Write higher order", write_higher_order);
      print_parameter(pcout, "Polynomial degree", degree);
    }
  }

  // set write_output = true in order to write files for visualization
  bool write_output;

  unsigned int output_counter_start;

  // output_folder
  std::string output_folder;

  // name of generated output files
  std::string output_name;

  // before then no output will be written
  double output_start_time;

  // specifies the time interval in which output is written
  double output_interval_time;

  // write higher order output (NOTE: requires at least ParaView version 5.5, switch off if ParaView
  // version is lower)
  bool write_higher_order;

  // defines polynomial degree used for output (for visualization in ParaView: Properties >
  // Miscellaneous > Nonlinear Subdivision Level (use a value > 1)) if write_higher_order = true. In
  // case of write_higher_order = false, this variable defines the number of subdivisions of a cell,
  // with ParaView using linear interpolation for visualization on these subdivided cells.
  unsigned int degree;
};

#endif /* INCLUDE_POSTPROCESSOR_OUTPUT_DATA_H_ */
