/*
 * write_output.cpp
 *
 *  Created on: May 16, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include "write_output.h"

namespace CompNS
{
template<int dim, typename Number, typename VectorType>
void
write_output(OutputData const &                              output_data,
             DoFHandler<dim> const &                         dof_handler,
             Mapping<dim> const &                            mapping,
             VectorType const &                              solution_conserved,
             std::vector<SolutionField<dim, Number>> const & additional_fields,
             unsigned int const                              output_counter)
{
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;
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

  std::ostringstream filename;
  filename << output_data.output_folder << output_data.output_name << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << "_" << output_counter << ".vtu";

  data_out.build_patches(mapping, output_data.degree, DataOut<dim>::curved_inner_cells);

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
                                    OutputData const &      output_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  output_data = output_data_in;

  // reset output counter
  output_counter = output_data.output_counter_start;
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::evaluate(
  VectorType const &                              solution_conserved,
  std::vector<SolutionField<dim, Number>> const & additional_fields,
  double const &                                  time,
  int const &                                     time_step_number)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

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
                                              output_counter);

        ++output_counter;
      }
    }
    else // steady problem (time_step_number = -1)
    {
      pcout << std::endl
            << "OUTPUT << Write " << (output_counter == 0 ? "initial" : "solution") << " data"
            << std::endl;

      write_output<dim, Number, VectorType>(
        output_data, *dof_handler, *mapping, solution_conserved, additional_fields, output_counter);

      ++output_counter;
    }
  }
}

template class OutputGenerator<2, float>;
template class OutputGenerator<2, double>;

template class OutputGenerator<3, float>;
template class OutputGenerator<3, double>;
} // namespace CompNS
