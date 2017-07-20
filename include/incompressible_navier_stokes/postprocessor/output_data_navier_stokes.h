/*
 * OutputDataNavierStokes.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_

#include "postprocessor/output_data.h"

struct OutputDataNavierStokes : public OutputData
{
  OutputDataNavierStokes()
    :
    write_divergence(false),
    write_velocity_magnitude(false),
    write_vorticity_magnitude(false),
    write_q_criterion(false)
  {}

  void print(ConditionalOStream &pcout, bool unsteady)
  {
    OutputData::print(pcout,unsteady);

    print_parameter(pcout,"Compute divergence",write_divergence);
  }

  // write divergence of velocity field
  bool write_divergence;

  // write velocity magnitude
  bool write_velocity_magnitude;

  // write vorticity magnitude
  bool write_vorticity_magnitude;

  // write Q criterion
  bool write_q_criterion;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_ */
