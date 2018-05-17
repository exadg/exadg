/*
 * OutputData.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_OUTPUT_DATA_H_
#define INCLUDE_POSTPROCESSOR_OUTPUT_DATA_H_

#include "../include/functionalities/print_functions.h"

using namespace dealii;

struct OutputData
{
  OutputData()
    :
    write_output(false),
    output_counter_start(0),
    output_folder("output"),
    output_name("name"),
    output_start_time(std::numeric_limits<double>::max()),
    output_interval_time(std::numeric_limits<double>::max()),
    number_of_patches(1)
  {}

  void print(ConditionalOStream &pcout, bool unsteady)
  {
    // output for visualization of results
    print_parameter(pcout,"Write output",write_output);
    if(write_output == true)
    {
      print_parameter(pcout,"Output counter start",output_counter_start);
      print_parameter(pcout,"Output folder",output_folder);
      print_parameter(pcout,"Name of output files",output_name);
      if(unsteady == true)
      {
        print_parameter(pcout,"Output start time",output_start_time);
        print_parameter(pcout,"Output interval time",output_interval_time);
      }
      print_parameter(pcout,"Number of patches",number_of_patches);
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

  // number of patches
  unsigned int number_of_patches;
};

#endif /* INCLUDE_POSTPROCESSOR_OUTPUT_DATA_H_ */
