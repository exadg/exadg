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
using namespace dealii;

template<int dim, typename Number, typename VectorType>
void
write_output(OutputData const &                              output_data,
             DoFHandler<dim> const &                         dof_handler,
             Mapping<dim> const &                            mapping,
             VectorType const &                              solution_conserved,
             std::vector<SolutionField<dim, Number>> const & additional_fields,
             unsigned int const                              output_counter,
             MPI_Comm const &                                mpi_comm)
{
  std::string folder = output_data.output_folder, file = output_data.output_name;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  // conserved variables
  std::vector<std::string> solution_names_conserved(dim + 2, "rho_u");
  solution_names_conserved[0]       = "rho";
  solution_names_conserved[1 + dim] = "rho_E";

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    solution_component_interpretation(dim + 2,
                                      DataComponentInterpretation::component_is_part_of_vector);

  solution_component_interpretation[0]       = DataComponentInterpretation::component_is_scalar;
  solution_component_interpretation[1 + dim] = DataComponentInterpretation::component_is_scalar;

  solution_conserved.update_ghost_values();
  data_out.add_data_vector(dof_handler,
                           solution_conserved,
                           solution_names_conserved,
                           solution_component_interpretation);

  // additional solution fields
  for(typename std::vector<SolutionField<dim, Number>>::const_iterator it =
        additional_fields.begin();
      it != additional_fields.end();
      ++it)
  {
    it->vector->update_ghost_values();

    if(it->type == SolutionFieldType::scalar)
    {
      data_out.add_data_vector(*it->dof_handler, *it->vector, it->name);
    }
    else if(it->type == SolutionFieldType::vector)
    {
      std::vector<std::string> names(dim, it->name);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(*it->dof_handler, *it->vector, names, component_interpretation);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  data_out.build_patches(mapping, output_data.degree, DataOut<dim>::curved_inner_cells);

  data_out.write_vtu_with_pvtu_record(folder, file, output_counter, mpi_comm, 4);
}

template<int dim, typename Number>
OutputGenerator<dim, Number>::OutputGenerator(MPI_Comm const & comm)
  : mpi_comm(comm), output_counter(0), reset_counter(true)
{
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::setup(DoFHandler<dim> const & dof_handler_in,
                                    Mapping<dim> const &    mapping_in,
                                    OutputData const &      output_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  output_data = output_data_in;

  // reset output counter
  output_counter = output_data.output_counter_start;

  if(output_data.write_output == true)
  {
    create_directories(output_data.output_folder, mpi_comm);

    // Visualize boundary IDs:
    // since boundary IDs typically do not change during the simulation, we only do this
    // once at the beginning of the simulation (i.e., in the setup function).
    if(output_data.write_boundary_IDs)
    {
      write_boundary_IDs(dof_handler->get_triangulation(),
                         output_data.output_folder,
                         output_data.output_name,
                         mpi_comm);
    }

    // write surface mesh
    if(output_data.write_surface_mesh)
    {
      write_surface_mesh(dof_handler->get_triangulation(),
                         *mapping,
                         output_data.degree,
                         output_data.output_folder,
                         output_data.output_name,
                         output_counter,
                         mpi_comm);
    }


    // processor_id
    if(output_data.write_processor_id)
    {
      GridOut grid_out;

      grid_out.write_mesh_per_processor_as_vtu(dof_handler->get_triangulation(),
                                               output_data.output_folder + output_data.output_name +
                                                 "_processor_id");
    }
  }
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::evaluate(
  VectorType const &                              solution_conserved,
  std::vector<SolutionField<dim, Number>> const & additional_fields,
  double const &                                  time,
  int const &                                     time_step_number)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  if(output_data.write_output == true)
  {
    if(time_step_number >= 0) // unsteady problem
    {
      // small number which is much smaller than the time step size
      double const EPSILON = 1.0e-10;

      // In the first time step, the current time might be larger than output_start_time. In that
      // case, we first have to reset the counter in order to avoid that output is written every
      // time step.
      if(reset_counter)
      {
        if(time > output_data.output_start_time)
        {
          output_counter += int((time - output_data.output_start_time + EPSILON) /
                                output_data.output_interval_time);
        }
        reset_counter = false;
      }


      if(time > (output_data.output_start_time + output_counter * output_data.output_interval_time -
                 EPSILON))
      {
        pcout << std::endl
              << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4)
              << time << std::endl;

        write_output<dim, Number, VectorType>(output_data,
                                              *dof_handler,
                                              *mapping,
                                              solution_conserved,
                                              additional_fields,
                                              output_counter,
                                              mpi_comm);

        ++output_counter;
      }
    }
    else // steady problem (time_step_number = -1)
    {
      pcout << std::endl
            << "OUTPUT << Write " << (output_counter == 0 ? "initial" : "solution") << " data"
            << std::endl;

      write_output<dim, Number, VectorType>(output_data,
                                            *dof_handler,
                                            *mapping,
                                            solution_conserved,
                                            additional_fields,
                                            output_counter,
                                            mpi_comm);

      ++output_counter;
    }
  }
}

template class OutputGenerator<2, float>;
template class OutputGenerator<2, double>;

template class OutputGenerator<3, float>;
template class OutputGenerator<3, double>;
} // namespace CompNS
} // namespace ExaDG
