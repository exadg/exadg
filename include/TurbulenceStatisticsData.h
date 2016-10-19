/*
 * TurbulenceStatisticsData.h
 *
 *  Created on: Oct 19, 2016
 *      Author: krank
 */

#ifndef INCLUDE_TURBULENCESTATISTICSDATA_H_
#define INCLUDE_TURBULENCESTATISTICSDATA_H_

struct TurbulenceStatisticsData
{
  TurbulenceStatisticsData()
    :
    statistics_start_time(1.e9),
    statistics_every(1),
    statistics_end_time(1.e9),
    viscosity(1.)
  {}

  // before then no statistics calculation will be performed
  double statistics_start_time;

  // calculate statistics every "statistics_every" time steps
  unsigned int statistics_every;

  // necessary for writing data just before end of simulation
  double statistics_end_time;

  // necessary for post-processing of turbulence quantities
  // does not have to be set as input parameter but is set in code
  double viscosity;

};

#endif /* INCLUDE_TURBULENCESTATISTICSDATA_H_ */
