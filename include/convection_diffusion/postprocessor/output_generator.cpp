/*
 * output_generator.cpp
 *
 *  Created on: May 14, 2019
 *      Author: fehn
 */

#include <deal.II/numerics/data_out.h>

#include "output_generator.h"

#include "../../postprocessor/write_output.h"

namespace ConvDiff
{
template<int dim, typename VectorType>
void
write_output(OutputDataBase const &  output_data,
             DoFHandler<dim> const & dof_handler,
             Mapping<dim> const &    mapping,
             VectorType const &      solution_vector,
             unsigned int const      output_counter)
{
  std::string folder = output_data.output_folder, file = output_data.output_name;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_vector, "solution");
  data_out.build_patches(mapping, output_data.degree, DataOut<dim>::curved_inner_cells);

  data_out.write_vtu_with_pvtu_record(folder, file, output_counter, 4);

  // write surface mesh
  if(output_data.write_surface_mesh)
  {
    write_surface_mesh(dof_handler.get_triangulation(),
                       mapping,
                       output_data.degree,
                       folder,
                       file + "_surface",
                       output_counter);
  }
}

template<int dim, typename Number>
OutputGenerator<dim, Number>::OutputGenerator() : output_counter(0), reset_counter(true)
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

  // Visualize boundary IDs:
  // since boundary IDs typically do not change during the simulation, we only do this
  // once at the beginning of the simulation (i.e., in the setup function).
  if(output_data.write_boundary_IDs)
  {
    write_boundary_IDs(dof_handler->get_triangulation(),
                       output_data.output_folder,
                       output_data.output_name);
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

        write_output<dim>(output_data, *dof_handler, *mapping, solution, output_counter);

        ++output_counter;
      }
    }
    else // steady problem (time_step_number = -1)
    {
      pcout << std::endl
            << "OUTPUT << Write " << (output_counter == 0 ? "initial" : "solution") << " data"
            << std::endl;

      write_output<dim>(output_data, *dof_handler, *mapping, solution, output_counter);

      ++output_counter;
    }
  }
}

template class OutputGenerator<2, float>;
template class OutputGenerator<3, float>;

template class OutputGenerator<2, double>;
template class OutputGenerator<3, double>;

} // namespace ConvDiff
