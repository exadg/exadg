/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_

#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

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

struct OutputData : public OutputDataBase
{
  OutputData()
    : write_vorticity(false),
      write_divergence(false),
      write_velocity_magnitude(false),
      write_vorticity_magnitude(false),
      write_streamfunction(false),
      write_q_criterion(false),
      mean_velocity(OutputDataMeanVelocity()),
      write_cfl(false),
      write_aspect_ratio(false)
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write vorticity", write_vorticity);
    print_parameter(pcout, "Write divergence", write_divergence);
    print_parameter(pcout, "Write velocity magnitude", write_velocity_magnitude);
    print_parameter(pcout, "Write vorticity magnitude", write_vorticity_magnitude);
    print_parameter(pcout, "Write streamfunction", write_streamfunction);
    print_parameter(pcout, "Write Q criterion", write_q_criterion);

    mean_velocity.print(pcout, unsteady);
  }

  // write vorticity of velocity field
  bool write_vorticity;

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

  // calculate mean velocity field (averaged over time)
  OutputDataMeanVelocity mean_velocity;

  // write cfl
  bool write_cfl;

  // write aspect ratio
  bool write_aspect_ratio;
};

template<int dim, typename Number>
class SpatialOperatorBase;

template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef SpatialOperatorBase<dim, Number> NavierStokesOperator;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(NavierStokesOperator const & navier_stokes_operator_in,
        DoFHandler<dim> const &      dof_handler_velocity_in,
        DoFHandler<dim> const &      dof_handler_pressure_in,
        Mapping<dim> const &         mapping_in,
        OutputData const &           output_data_in);

  void
  evaluate(VectorType const & velocity,
           VectorType const & pressure,
           double const &     time,
           int const &        time_step_number);

private:
  void
  initialize_additional_fields();

  void
  compute_mean_velocity(VectorType &       mean_velocity,
                        VectorType const & velocity,
                        double const       time,
                        int const          time_step_number);

  void
  calculate_additional_fields(VectorType const & velocity,
                              double const &     time,
                              int const &        time_step_number);

  MPI_Comm const mpi_comm;

  unsigned int output_counter;
  bool         reset_counter;

  OutputData output_data;

  SmartPointer<DoFHandler<dim> const>      dof_handler_velocity;
  SmartPointer<DoFHandler<dim> const>      dof_handler_pressure;
  SmartPointer<Mapping<dim> const>         mapping;
  SmartPointer<NavierStokesOperator const> navier_stokes_operator;

  // additional fields
  VectorType   vorticity;
  VectorType   divergence;
  VectorType   velocity_magnitude;
  VectorType   vorticity_magnitude;
  VectorType   streamfunction;
  VectorType   q_criterion;
  VectorType   mean_velocity; // velocity field averaged over time
  VectorType   cfl;
  unsigned int counter_mean_velocity;

  std::vector<SolutionField<dim, Number>> additional_fields;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
