/*
 * turbulence_statistics_data.h
 *
 *  Created on: Oct 19, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TURBULENCE_STATISTICS_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TURBULENCE_STATISTICS_DATA_H_

struct TurbulenceStatisticsData
{
  TurbulenceStatisticsData()
    : statistics_start_time(1.e9),
      statistics_every(1),
      statistics_end_time(1.e9),
      viscosity(1.),
      write_output_q_criterion(false)
  {
  }

  // before then no statistics calculation will be performed
  double statistics_start_time;

  // calculate statistics every "statistics_every" time steps
  unsigned int statistics_every;

  // necessary for writing data just before end of simulation
  double statistics_end_time;

  // necessary for post-processing of turbulence quantities
  // does not have to be set as input parameter but is set in code
  double viscosity;

  bool write_output_q_criterion;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_TURBULENCE_STATISTICS_DATA_H_ */
