/*
 * output_data_navier_stokes.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_

#include "../../postprocessor/output_data.h"

/*
 *  Average velocity field over time for statistically steady, turbulent
 *  flow problems in order to visualize the time-averaged velocity field.
 *  Of course, this module can also be used for statistically unsteady problems,
 *  but in this case the mean velocity field is probably not meaningful.
 */
struct OutputDataMeanVelocity
{
  OutputDataMeanVelocity()
    : calculate(false), sample_start_time(0.0), sample_end_time(0.0), sample_every_timesteps(1)
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    if(unsteady)
    {
      print_parameter(pcout, "Calculate mean velocity", calculate);
      print_parameter(pcout, "  Sample start time", sample_start_time);
      print_parameter(pcout, "  Sample end time", sample_end_time);
      print_parameter(pcout, "  Sample every timesteps", sample_every_timesteps);
    }
  }

  // calculate mean velocity field (for statistically steady, turbulent flows)
  bool calculate;

  // sampling information
  double       sample_start_time;
  double       sample_end_time;
  unsigned int sample_every_timesteps;
};

struct OutputDataNavierStokes : public OutputData
{
  OutputDataNavierStokes()
    : write_divergence(false),
      write_velocity_magnitude(false),
      write_vorticity_magnitude(false),
      write_streamfunction(false),
      write_q_criterion(false),
      write_processor_id(false),
      mean_velocity(OutputDataMeanVelocity())
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    OutputData::print(pcout, unsteady);

    print_parameter(pcout, "Write divergence", write_divergence);
    print_parameter(pcout, "Write velocity magnitude", write_velocity_magnitude);
    print_parameter(pcout, "Write vorticity magnitude", write_vorticity_magnitude);
    print_parameter(pcout, "Write streamfunction", write_streamfunction);
    print_parameter(pcout, "Write Q criterion", write_q_criterion);
    print_parameter(pcout, "Write processor ID", write_processor_id);

    mean_velocity.print(pcout, unsteady);
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

  // calculate mean velocity field (averaged over time)
  OutputDataMeanVelocity mean_velocity;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_DATA_NAVIER_STOKES_H_ */
