/*
 * output_generator.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

// deal.II
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

// ExaDG
#include <exadg/postprocessor/write_output.h>
#include <exadg/structure/postprocessor/output_generator.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename VectorType>
void
write_output(OutputDataBase const &  output_data,
             DoFHandler<dim> const & dof_handler,
             Mapping<dim> const &    mapping,
             VectorType const &      solution_vector,
             unsigned int const      output_counter,
             MPI_Comm const &        mpi_comm)
{
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  std::vector<std::string>                                              names(dim, "displacement");
  std::vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation(
    dim, DataComponentInterpretation::component_is_part_of_vector);

  solution_vector.update_ghost_values();
  data_out.add_data_vector(dof_handler, solution_vector, names, component_interpretation);

  data_out.build_patches(mapping, output_data.degree, DataOut<dim>::curved_inner_cells);

  data_out.write_vtu_with_pvtu_record(
    output_data.output_folder, output_data.output_name, output_counter, mpi_comm, 4);
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
                                    OutputDataBase const &  output_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  output_data = output_data_in;

  // reset output counter
  output_counter = output_data.output_counter_start;

  if(output_data.write_output == true)
  {
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
                         0,
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
OutputGenerator<dim, Number>::evaluate(VectorType const & solution,
                                       double const &     time,
                                       int const &        time_step_number)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  if(output_data.write_output == true)
  {
    if(time_step_number >= 0) // unsteady problem
    {
      // small number which is much smaller than the time step size
      const double EPSILON = 1.0e-10;

      // The current time might be larger than output_start_time. In that case, we first have to
      // reset the counter in order to avoid that output is written every time step.
      if(reset_counter)
      {
        output_counter +=
          int((time - output_data.output_start_time + EPSILON) / output_data.output_interval_time);
        reset_counter = false;
      }

      if(time > (output_data.output_start_time + output_counter * output_data.output_interval_time -
                 EPSILON))
      {
        pcout << std::endl
              << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4)
              << time << std::endl;

        write_output<dim>(output_data, *dof_handler, *mapping, solution, output_counter, mpi_comm);

        ++output_counter;
      }
    }
    else // steady problem (time_step_number = -1)
    {
      pcout << std::endl
            << "OUTPUT << Write " << (output_counter == 0 ? "initial" : "solution") << " data"
            << std::endl;

      write_output<dim>(output_data, *dof_handler, *mapping, solution, output_counter, mpi_comm);

      ++output_counter;
    }
  }
}

template class OutputGenerator<2, float>;
template class OutputGenerator<3, float>;

template class OutputGenerator<2, double>;
template class OutputGenerator<3, double>;

} // namespace Structure
} // namespace ExaDG
