/*
 * output_generator.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

// deal.II
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include "output_generator.h"

namespace Structure
{
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
OutputGenerator<dim, Number>::setup(DoFHandler<dim> const & dof_handler,
                                    Mapping<dim> const &    mapping,
                                    OutputDataBase const &  output_data)
{
  this->dof_handler = &dof_handler;
  this->mapping     = &mapping;
  this->output_data = output_data;

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
