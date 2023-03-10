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
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace IncNS
{
struct OutputData : public OutputDataBase
{
  OutputData()
    : write_vorticity(false),
      write_divergence(false),
      write_shear_rate(false),
      write_velocity_magnitude(false),
      write_vorticity_magnitude(false),
      write_wall_shear_stress(false),
      write_wall_shear_stress_boundary_IDs({{dealii::numbers::invalid_boundary_id}}),
      write_streamfunction(false),
      write_q_criterion(false),
      mean_velocity(TimeControlData()),
      write_cfl(false),
      write_aspect_ratio(false)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write vorticity", write_vorticity);
    print_parameter(pcout, "Write divergence", write_divergence);
    print_parameter(pcout, "Write shear rate", write_shear_rate);
    print_parameter(pcout, "Write velocity magnitude", write_velocity_magnitude);
    print_parameter(pcout, "Write vorticity magnitude", write_vorticity_magnitude);
    print_parameter(pcout, "Write wall shear stress", write_wall_shear_stress);
    print_parameter(pcout, "Write streamfunction", write_streamfunction);
    print_parameter(pcout, "Write Q criterion", write_q_criterion);

    mean_velocity.print(pcout, unsteady);
  }

  // write vorticity of velocity field
  bool write_vorticity;

  // write divergence of velocity field
  bool write_divergence;

  // write shear rate in velocity field
  bool write_shear_rate;

  // write velocity magnitude
  bool write_velocity_magnitude;

  // write vorticity magnitude
  bool write_vorticity_magnitude;

  // write wall shear stress on IDs
  bool                                 write_wall_shear_stress;
  std::set<dealii::types::boundary_id> write_wall_shear_stress_boundary_IDs;

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

  //  Average velocity field over time for statistically steady, turbulent
  //  flow problems in order to visualize the time-averaged velocity field.
  //  Of course, this module can also be used for statistically unsteady problems,
  //  but in this case the mean velocity field is probably not meaningful.
  TimeControlData mean_velocity;


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
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler_velocity_in,
        dealii::DoFHandler<dim> const & dof_handler_pressure_in,
        dealii::Mapping<dim> const &    mapping_in,
        OutputData const &              output_data_in);

  void
  evaluate(VectorType const &                                                    velocity,
           VectorType const &                                                    pressure,
           std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> const & additional_fields,
           std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> const & surface_fields,
           double const                                                          time,
           bool const                                                            unsteady);

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  OutputData output_data;

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_velocity;
  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_pressure;
  dealii::SmartPointer<dealii::Mapping<dim> const>    mapping;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
