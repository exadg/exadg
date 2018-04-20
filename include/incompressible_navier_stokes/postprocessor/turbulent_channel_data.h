/*
 * turbulent_channel_data.h
 *
 *  Created on: Apr 20, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TURBULENT_CHANNEL_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TURBULENT_CHANNEL_DATA_H_


// turbulent channel data

struct TurbulentChannelData
{
  TurbulentChannelData()
   :
   calculate_statistics(false),
   cells_are_stretched(false),
   sample_start_time(0.0),
   sample_end_time(1.0),
   sample_every_timesteps(1),
   viscosity(1.0),
   density(1.0),
   filename_prefix("indexa")
  {}

  void print(ConditionalOStream &pcout)
  {
    if(calculate_statistics == true)
    {
      pcout << "  Turbulent channel statistics:" << std::endl;
      print_parameter(pcout,"Calculate statistics",calculate_statistics);
      print_parameter(pcout,"Cells are stretched",cells_are_stretched);
      print_parameter(pcout,"Sample start time",sample_start_time);
      print_parameter(pcout,"Sample end time",sample_end_time);
      print_parameter(pcout,"Sample every timesteps",sample_every_timesteps);
      print_parameter(pcout,"Dynamic viscosity",viscosity);
      print_parameter(pcout,"Density",density);
      print_parameter(pcout,"Filename prefix",filename_prefix);
    }
  }

  // calculate statistics?
  bool calculate_statistics;

  // are cells stretched, i.e., is a volume manifold applied?
  bool cells_are_stretched;

  // start time for sampling
  double sample_start_time;

  // end time for sampling
  double sample_end_time;

  // perform sampling every ... timesteps
  unsigned int sample_every_timesteps;

  // dynamic viscosity
  double viscosity;

  // density
  double density;

  std::string filename_prefix;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TURBULENT_CHANNEL_DATA_H_ */
