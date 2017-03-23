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
    compute_divergence(false)
  {}

  void print(ConditionalOStream &pcout, bool unsteady)
  {
    OutputData::print(pcout,unsteady);

    print_parameter(pcout,"Compute divergence",compute_divergence);
  }

  // compute divergence of velocity field
  bool compute_divergence;

};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_ */
