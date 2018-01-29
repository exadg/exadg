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
    write_streamfunction(false),
    write_q_criterion(false),
    write_processor_id(false)
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

  // Calculate streamfunction in order to visualize streamlines!
  // Note that this option is only available in 2D!
  // To calculate the streamfunction Psi, a Poisson equation is solved
  // with homogeneous Dirichlet BC's. Accordingly, this approach can only be
  // used for flow problems where the whole boundary is one streamline, e.g.,
  // cavity-type flow problems where the velocity is tangential to the boundary
  // on one part of the boundary and 0 (no-slip) on the rest of the boundary.
  bool write_streamfunction;

  // write Q criterion
  bool write_q_criterion;

  // write processor ID to scalar field in order to visualize the
  // distribution of cells to processors
  bool write_processor_id;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_ */
