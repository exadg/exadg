/*
 * output_generator.h
 *
 *  Created on: Mar 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_OUTPUT_GENERATOR_H_

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

namespace ConvDiff
{

template<int dim>
void write_output(OutputData const                            &output_data,
                  DoFHandler<dim> const                       &dof_handler,
                  Mapping<dim> const                          &mapping,
                  parallel::distributed::Vector<double> const &solution_vector,
                  unsigned int const                          output_counter)
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution_vector, "solution");
  data_out.build_patches (output_data.number_of_patches);

  const std::string filename = output_data.output_folder + output_data.output_name + "_" + Utilities::int_to_string (output_counter, 3);

  data_out.build_patches (mapping, output_data.number_of_patches, DataOut<dim>::curved_inner_cells);

  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);
}

template<int dim>
class OutputGenerator
{
public:
  OutputGenerator()
    :
    output_counter(0)
  {}

  void setup(DoFHandler<dim> const &dof_handler_in,
             Mapping<dim> const    &mapping_in,
             OutputData const      &output_data_in)
  {
    dof_handler = &dof_handler_in;
    mapping = &mapping_in;
    output_data = output_data_in;

    // reset output counter
    output_counter = output_data.output_counter_start;
  }

  void evaluate(parallel::distributed::Vector<double> const   &solution,
                double const                                  &time,
                int const                                     &time_step_number)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    if(output_data.write_output == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
        if(time > (output_data.output_start_time + output_counter*output_data.output_interval_time - EPSILON))
        {
          pcout << std::endl << "OUTPUT << Write data at time t = "
                << std::scientific << std::setprecision(4) << time << std::endl;

          write_output<dim>(output_data,
                            *dof_handler,
                            *mapping,
                            solution,
                            output_counter);

          ++output_counter;
        }
      }
      else // steady problem (time_step_number = -1)
      {
        pcout << std::endl << "OUTPUT << Write "
              << (output_counter == 0 ? "initial" : "solution") << " data"
              << std::endl;

        write_output<dim>(output_data,
                          *dof_handler,
                          *mapping,
                          solution,
                          output_counter);

        ++output_counter;
      }
    }
  }

private:
  unsigned int output_counter;

  SmartPointer< DoFHandler<dim> const > dof_handler;
  SmartPointer< Mapping<dim> const > mapping;
  OutputData output_data;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
