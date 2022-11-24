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

// deal.II
#include <deal.II/numerics/data_out.h>

// ExaDG
#include <exadg/compressible_navier_stokes/postprocessor/output_generator.h>
#include <exadg/postprocessor/write_output.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim, typename Number, typename VectorType>
void
write_output(
  OutputData const &                                                    output_data,
  dealii::DoFHandler<dim> const &                                       dof_handler,
  dealii::Mapping<dim> const &                                          mapping,
  VectorType const &                                                    solution_conserved,
  std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> const & additional_fields,
  unsigned int const                                                    output_counter,
  MPI_Comm const &                                                      mpi_comm)
{
  std::string folder = output_data.directory, file = output_data.filename;

  dealii::DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;

  dealii::DataOut<dim> data_out;
  data_out.set_flags(flags);

  // conserved variables
  std::vector<std::string> solution_names_conserved(dim + 2, "rho_u");
  solution_names_conserved[0]       = "rho";
  solution_names_conserved[1 + dim] = "rho_E";

  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    solution_component_interpretation(
      dim + 2, dealii::DataComponentInterpretation::component_is_part_of_vector);

  solution_component_interpretation[0] = dealii::DataComponentInterpretation::component_is_scalar;
  solution_component_interpretation[1 + dim] =
    dealii::DataComponentInterpretation::component_is_scalar;

  data_out.add_data_vector(dof_handler,
                           solution_conserved,
                           solution_names_conserved,
                           solution_component_interpretation);

  // additional solution fields
  for(auto & additional_field : additional_fields)
  {
    if(additional_field->get_type() == SolutionFieldType::scalar)
    {
      data_out.add_data_vector(additional_field->get_dof_handler(),
                               additional_field->get(),
                               additional_field->get_name());
    }
    else if(additional_field->get_type() == SolutionFieldType::vector)
    {
      std::vector<std::string> names(dim, additional_field->get_name());
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation(dim,
                                 dealii::DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(additional_field->get_dof_handler(),
                               additional_field->get(),
                               names,
                               component_interpretation);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
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
                         time_control.get_counter(),
                         mpi_comm);
    }

    // write grid
    if(output_data.write_grid)
    {
      write_grid(dof_handler->get_triangulation(), output_data.directory, output_data.filename);
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

  write_output<dim, Number, VectorType>(output_data,
                                        *dof_handler,
                                        *mapping,
                                        solution_conserved,
                                        additional_fields,
                                        time_control.get_counter(),
                                        mpi_comm);
}


template class OutputGenerator<2, float>;
template class OutputGenerator<2, double>;

template class OutputGenerator<3, float>;
template class OutputGenerator<3, double>;
} // namespace CompNS
} // namespace ExaDG
