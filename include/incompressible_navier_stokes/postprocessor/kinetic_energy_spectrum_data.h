/*
 * kinetic_energy_spectrum_data.h
 *
 *  Created on: Mar 15, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_DATA_H_


// kinetic energy spectrum data

struct KineticEnergySpectrumData
{
  KineticEnergySpectrumData()
    : calculate(false),
      start_time(0.0),
      calculate_every_time_steps(-1),
      calculate_every_time_interval(-1.0),
      filename_prefix("energy_spectrum"),
      output_tolerance(std::numeric_limits<double>::min()),
      evaluation_points_per_cell(0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << std::endl << "  Calculate kinetic energy spectrum:" << std::endl;
      print_parameter(pcout, "Start time", start_time);
      if(calculate_every_time_steps >= 0)
        print_parameter(pcout, "Calculate every timesteps", calculate_every_time_steps);
      if(calculate_every_time_interval >= 0.0)
        print_parameter(pcout, "Calculate every time interval", calculate_every_time_interval);
      print_parameter(pcout, "Output precision", output_tolerance);
      print_parameter(pcout, "Evaluation points per cell", evaluation_points_per_cell);
    }
  }

  bool         calculate;
  double       start_time;
  int          calculate_every_time_steps;
  double       calculate_every_time_interval;
  std::string  filename_prefix;
  double       output_tolerance;
  unsigned int evaluation_points_per_cell;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_DATA_H_ */
