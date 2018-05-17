/*
 * WriteOutputNavierStokes.h
 *
 *  Created on: Oct 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_WRITE_OUTPUT_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_WRITE_OUTPUT_NAVIER_STOKES_H_

#include <deal.II/numerics/data_out.h>

#include "../../incompressible_navier_stokes/postprocessor/output_data_navier_stokes.h"

template<int dim, typename Number>
void write_output(OutputDataNavierStokes const                &output_data,
                  DoFHandler<dim> const                       &dof_handler_velocity,
                  DoFHandler<dim> const                       &dof_handler_pressure,
                  Mapping<dim> const                          &mapping,
                  parallel::distributed::Vector<Number> const &velocity,
                  parallel::distributed::Vector<Number> const &pressure,
                  parallel::distributed::Vector<Number> const &vorticity,
                  std::vector<SolutionField<dim,Number> > const &additional_fields,
                  unsigned int const                          output_counter)
{
  DataOut<dim> data_out;

  std::vector<std::string> velocity_names (dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    velocity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (dof_handler_velocity, velocity, velocity_names, velocity_component_interpretation);

  std::vector<std::string> vorticity_names (dim, "vorticity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vorticity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (dof_handler_velocity, vorticity, vorticity_names, vorticity_component_interpretation);

  pressure.update_ghost_values();
  data_out.add_data_vector (dof_handler_pressure, pressure, "p");

  for(typename std::vector<SolutionField<dim,Number> >::const_iterator
      it = additional_fields.begin(); it!=additional_fields.end(); ++it)
  {
    if(it->type == SolutionFieldType::scalar)
    {
      data_out.add_data_vector (*it->dof_handler, *it->vector, it->name);
    }
    else if(it->type == SolutionFieldType::vector)
    {
      std::vector<std::string> names (dim, it->name);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector (*it->dof_handler, *it->vector, names, component_interpretation);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  std::ostringstream filename;
  filename << output_data.output_folder
           << output_data.output_name
           << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           << "_"
           << output_counter
           << ".vtu";

  data_out.build_patches (mapping, output_data.number_of_patches, DataOut<dim>::curved_inner_cells);

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
    {
      std::ostringstream filename;
      filename << output_data.output_name
               << "_Proc"
               << i
               << "_"
               << output_counter
               << ".vtu";

      filenames.push_back(filename.str().c_str());
    }
    std::string master_name = output_data.output_folder + output_data.output_name + "_" + Utilities::int_to_string(output_counter) + ".pvtu";
    std::ofstream master_output (master_name.c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
}

template<int dim, typename Number>
class OutputGenerator
{
public:
  OutputGenerator()
    :
    output_counter(0)
  {}

  void setup(DoFHandler<dim> const        &dof_handler_velocity_in,
             DoFHandler<dim> const        &dof_handler_pressure_in,
             Mapping<dim> const           &mapping_in,
             OutputDataNavierStokes const &output_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    output_data = output_data_in;

    // reset output counter
    output_counter = output_data.output_counter_start;
  }

  void evaluate(parallel::distributed::Vector<Number> const   &velocity,
                parallel::distributed::Vector<Number> const   &pressure,
                parallel::distributed::Vector<Number> const   &vorticity,
                std::vector<SolutionField<dim,Number> > const &additional_fields,
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
                            *dof_handler_velocity,
                            *dof_handler_pressure,
                            *mapping,
                            velocity,
                            pressure,
                            vorticity,
                            additional_fields,
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
                          *dof_handler_velocity,
                          *dof_handler_pressure,
                          *mapping,
                          velocity,
                          pressure,
                          vorticity,
                          additional_fields,
                          output_counter);

        ++output_counter;
      }
    }
  }

private:
  unsigned int output_counter;

  SmartPointer< DoFHandler<dim> const > dof_handler_velocity;
  SmartPointer< DoFHandler<dim> const > dof_handler_pressure;
  SmartPointer< Mapping<dim> const > mapping;
  OutputDataNavierStokes output_data;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_WRITE_OUTPUT_NAVIER_STOKES_H_ */
