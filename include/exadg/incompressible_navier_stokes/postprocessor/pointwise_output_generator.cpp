/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/incompressible_navier_stokes/postprocessor/pointwise_output_generator.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim>
PointwiseOutputData<dim>::PointwiseOutputData() : write_velocity(false), write_pressure(false)
{
}

template<int dim>
void
PointwiseOutputData<dim>::print(dealii::ConditionalOStream & pcout) const
{
  PointwiseOutputDataBase<dim>::print(pcout);

  if(this->time_control_data.is_active && this->evaluation_points.size() > 0)
  {
    print_parameter(pcout, "Write velocity", write_velocity);
    print_parameter(pcout, "Write pressure", write_pressure);
  }
}

template struct PointwiseOutputData<2>;
template struct PointwiseOutputData<3>;

template<int dim, typename Number>
PointwiseOutputGenerator<dim, Number>::PointwiseOutputGenerator(MPI_Comm const & comm)
  : PointwiseOutputGeneratorBase<dim, Number>(comm)
{
}

template<int dim, typename Number>
void
PointwiseOutputGenerator<dim, Number>::setup(
  dealii::DoFHandler<dim> const &  dof_handler_velocity_in,
  dealii::DoFHandler<dim> const &  dof_handler_pressure_in,
  dealii::Mapping<dim> const &     mapping_in,
  PointwiseOutputData<dim> const & pointwise_output_data_in)
{
  this->setup_base(dof_handler_pressure_in.get_triangulation(),
                   mapping_in,
                   pointwise_output_data_in);

  dof_handler_velocity = &dof_handler_velocity_in;
  dof_handler_pressure = &dof_handler_pressure_in;

  pointwise_output_data = pointwise_output_data_in;

  if(pointwise_output_data.time_control_data.is_active and
     pointwise_output_data.evaluation_points.size() > 0)
  {
    if(pointwise_output_data.write_velocity)
      this->add_quantity("Velocity", dim);
    if(pointwise_output_data.write_pressure)
      this->add_quantity("Pressure", 1);
  }
}

template<int dim, typename Number>
void
PointwiseOutputGenerator<dim, Number>::evaluate(VectorType const & velocity,
                                                VectorType const & pressure,
                                                double const       time,
                                                bool const         unsteady)
{
  this->do_evaluate(
    [&]() {
      if(pointwise_output_data.write_velocity)
      {
        auto const values =
          this->template compute_point_values<dim>(velocity, *dof_handler_velocity);
        this->write_quantity("Velocity", values, 0 /*first_selected_component*/);
      }
      if(pointwise_output_data.write_pressure)
      {
        auto const values = this->template compute_point_values<1>(pressure, *dof_handler_pressure);
        this->write_quantity("Pressure", values);
      }
    },
    time,
    unsteady);
}

template class PointwiseOutputGenerator<2, float>;
template class PointwiseOutputGenerator<2, double>;

template class PointwiseOutputGenerator<3, float>;
template class PointwiseOutputGenerator<3, double>;

} // namespace IncNS
} // namespace ExaDG
