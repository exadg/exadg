/*
 * output_generator.cpp
 *
 *  Created on: May 14, 2019
 *      Author: fehn
 */

#include "output_generator.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

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
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;
  data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_vector, "solution");
  data_out.build_patches(output_data.degree);

  data_out.build_patches(mapping, output_data.degree, DataOut<dim>::curved_inner_cells);

  std::ostringstream filename;
  filename << output_data.output_folder << output_data.output_name << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << "_" << output_counter << ".vtu";

  std::ofstream output(filename.str().c_str());
  data_out.write_vtu(output);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
    {
      std::ostringstream filename;
      filename << output_data.output_name << "_Proc" << i << "_" << output_counter << ".vtu";

      filenames.push_back(filename.str().c_str());
    }
    std::string master_name = output_data.output_folder + output_data.output_name + "_" +
                              Utilities::int_to_string(output_counter) + ".pvtu";

    std::ofstream master_output(master_name.c_str());
    data_out.write_pvtu_record(master_output, filenames);
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
