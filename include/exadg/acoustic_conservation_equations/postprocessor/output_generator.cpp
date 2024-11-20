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

// C/C++
#include <fstream>

// deal.II
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>

// ExaDG
#include <exadg/acoustic_conservation_equations/postprocessor/output_generator.h>
#include <exadg/grid/grid_data.h>
#include <exadg/postprocessor/write_output.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim, typename Number>
void
write_output(OutputData const &                                         output_data,
             dealii::DoFHandler<dim> const &                            dof_handler_pressure,
             dealii::DoFHandler<dim> const &                            dof_handler_velocity,
             dealii::Mapping<dim> const &                               mapping,
             dealii::LinearAlgebra::distributed::Vector<Number> const & pressure,
             dealii::LinearAlgebra::distributed::Vector<Number> const & velocity,
             unsigned int const                                         output_counter,
             MPI_Comm const &                                           mpi_comm)
{
  std::string folder = output_data.directory, file = output_data.filename;

  dealii::DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;

  dealii::DataOut<dim> data_out;
  data_out.set_flags(flags);

  if(output_data.write_pressure)
    data_out.add_data_vector(dof_handler_pressure, pressure, "p");

  if(output_data.write_velocity)
  {
    std::vector<std::string> velocity_names(dim, "velocity_times_density");
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      velocity_component_interpretation(
        dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(dof_handler_velocity,
                             velocity,
                             velocity_names,
                             velocity_component_interpretation);
  }

  data_out.build_patches(mapping, output_data.degree, dealii::DataOut<dim>::curved_inner_cells);

  data_out.write_vtu_with_pvtu_record(folder, file, output_counter, mpi_comm, 4);
}

template<int dim, typename Number>
OutputGenerator<dim, Number>::OutputGenerator(MPI_Comm const & comm) : mpi_comm(comm)
{
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::setup(dealii::DoFHandler<dim> const & dof_handler_pressure_in,
                                    dealii::DoFHandler<dim> const & dof_handler_velocity_in,
                                    dealii::Mapping<dim> const &    mapping_in,
                                    OutputData const &              output_data_in)
{
  dof_handler_velocity = &dof_handler_velocity_in;
  dof_handler_pressure = &dof_handler_pressure_in;
  mapping              = &mapping_in;
  output_data          = output_data_in;

  time_control.setup(output_data_in.time_control_data);

  if(output_data.time_control_data.is_active)
  {
    create_directories(output_data.directory, mpi_comm);

    // Visualize boundary IDs:
    // since boundary IDs typically do not change during the simulation, we only do this
    // once at the beginning of the simulation (i.e., in the setup function).
    if(output_data.write_boundary_IDs)
    {
      write_boundary_IDs(dof_handler_pressure->get_triangulation(),
                         output_data.directory,
                         output_data.filename,
                         mpi_comm);
    }

    // write surface mesh
    if(output_data.write_surface_mesh)
    {
      write_surface_mesh(dof_handler_pressure->get_triangulation(),
                         *mapping,
                         output_data.degree,
                         output_data.directory,
                         output_data.filename,
                         0,
                         mpi_comm);
    }

    // write grid
    if(output_data.write_grid)
    {
      write_grid(dof_handler_pressure->get_triangulation(),
                 *mapping,
                 output_data.degree,
                 output_data.directory,
                 output_data.filename,
                 0,
                 mpi_comm);
    }

    // processor_id
    if(output_data.write_processor_id)
    {
      dealii::GridOut grid_out;

      grid_out.write_mesh_per_processor_as_vtu(dof_handler_pressure->get_triangulation(),
                                               output_data.directory + output_data.filename +
                                                 "_processor_id");
    }
  }
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::evaluate(VectorType const & pressure,
                                       VectorType const & velocity,
                                       double const       time,
                                       bool const         unsteady) const
{
  print_write_output_time(time, time_control.get_counter(), unsteady, mpi_comm);

  write_output<dim>(output_data,
                    *dof_handler_pressure,
                    *dof_handler_velocity,
                    *mapping,
                    pressure,
                    velocity,
                    time_control.get_counter(),
                    mpi_comm);
}

template class OutputGenerator<2, float>;
template class OutputGenerator<2, double>;

template class OutputGenerator<3, float>;
template class OutputGenerator<3, double>;

} // namespace Acoustics
} // namespace ExaDG
