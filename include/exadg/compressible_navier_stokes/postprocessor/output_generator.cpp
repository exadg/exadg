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

// ExaDG
#include <exadg/compressible_navier_stokes/postprocessor/output_generator.h>
#include <exadg/postprocessor/write_output.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim, typename Number>
OutputGenerator<dim, Number>::OutputGenerator(MPI_Comm const & comm) : mpi_comm(comm)
{
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::setup(dealii::DoFHandler<dim> const & dof_handler_in,
                                    dealii::Mapping<dim> const &    mapping_in,
                                    OutputData const &              output_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  output_data = output_data_in;

  time_control.setup(output_data_in.time_control_data);

  if(output_data_in.time_control_data.is_active)
  {
    create_directories(output_data.directory, mpi_comm);

    // Visualize boundary IDs:
    // since boundary IDs typically do not change during the simulation, we only do this
    // once at the beginning of the simulation (i.e., in the setup function).
    if(output_data.write_boundary_IDs)
    {
      write_boundary_IDs(dof_handler->get_triangulation(),
                         output_data.directory,
                         output_data.filename,
                         mpi_comm);
    }

    // write surface mesh
    if(output_data.write_surface_mesh)
    {
      write_surface_mesh(dof_handler->get_triangulation(),
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
      write_grid(dof_handler->get_triangulation(),
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

      grid_out.write_mesh_per_processor_as_vtu(dof_handler->get_triangulation(),
                                               output_data.directory + output_data.filename +
                                                 "_processor_id");
    }
  }
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::evaluate(
  VectorType const &                                                    solution_conserved,
  std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> const & additional_fields,
  double const                                                          time,
  bool const                                                            unsteady)
{
  print_write_output_time(time, time_control.get_counter(), unsteady, mpi_comm);

  VectorWriter<dim, Number> vector_writer(output_data, time_control.get_counter(), mpi_comm);

  std::vector<std::string> component_names(dim + 2, "rho_u");
  component_names[0]       = "rho";
  component_names[dim + 1] = "rho_E";

  std::vector<bool> component_is_part_of_vector(dim + 2, true);
  component_is_part_of_vector[0]       = false;
  component_is_part_of_vector[dim + 1] = false;

  vector_writer.add_data_vector(solution_conserved,
                                *dof_handler,
                                component_names,
                                component_is_part_of_vector);

  vector_writer.add_fields(additional_fields);

  vector_writer.write_aspect_ratio(*dof_handler, *mapping);

  vector_writer.write_pvtu(&(*mapping));
}


template class OutputGenerator<2, float>;
template class OutputGenerator<2, double>;

template class OutputGenerator<3, float>;
template class OutputGenerator<3, double>;
} // namespace CompNS
} // namespace ExaDG
