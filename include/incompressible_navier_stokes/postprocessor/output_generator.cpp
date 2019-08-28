/*
 * output_generator.cpp
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

#include <deal.II/numerics/data_out.h>

#include "output_generator.h"

#include "../spatial_discretization/dg_navier_stokes_base.h"

namespace IncNS
{
template<int dim, typename Number>
void
write_output(OutputData const &                                 output_data,
             DoFHandler<dim> const &                            dof_handler_velocity,
             DoFHandler<dim> const &                            dof_handler_pressure,
             Mapping<dim> const &                               mapping,
             LinearAlgebra::distributed::Vector<Number> const & velocity,
             LinearAlgebra::distributed::Vector<Number> const & pressure,
             std::vector<SolutionField<dim, Number>> const &    additional_fields,
             unsigned int const                                 output_counter)
{
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = output_data.write_higher_order;
  data_out.set_flags(flags);

  std::vector<std::string> velocity_names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    velocity_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);

  data_out.add_data_vector(dof_handler_velocity,
                           velocity,
                           velocity_names,
                           velocity_component_interpretation);

  pressure.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pressure, "p");

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
OutputGenerator<dim, Number>::OutputGenerator()
  : output_counter(0), reset_counter(true), counter_mean_velocity(0)
{
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::setup(NavierStokesOperator const & navier_stokes_operator_in,
                                    DoFHandler<dim> const &      dof_handler_velocity_in,
                                    DoFHandler<dim> const &      dof_handler_pressure_in,
                                    Mapping<dim> const &         mapping_in,
                                    OutputData const &           output_data_in)
{
  navier_stokes_operator = &navier_stokes_operator_in;
  dof_handler_velocity   = &dof_handler_velocity_in;
  dof_handler_pressure   = &dof_handler_pressure_in;
  mapping                = &mapping_in;
  output_data            = output_data_in;

  // reset output counter
  output_counter = output_data.output_counter_start;

  initialize_additional_fields();
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::evaluate(VectorType const & velocity,
                                       VectorType const & pressure,
                                       double const &     time,
                                       int const &        time_step_number)
{
  if(output_data.write_output == true)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

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

        calculate_additional_fields(velocity, time, time_step_number);

        write_output<dim>(output_data,
                          *dof_handler_velocity,
                          *dof_handler_pressure,
                          *mapping,
                          velocity,
                          pressure,
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

      calculate_additional_fields(velocity, time, time_step_number);

      write_output<dim>(output_data,
                        *dof_handler_velocity,
                        *dof_handler_pressure,
                        *mapping,
                        velocity,
                        pressure,
                        additional_fields,
                        output_counter);

      ++output_counter;
    }
  }
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::initialize_additional_fields()
{
  if(output_data.write_output == true)
  {
    // vorticity
    if(output_data.write_vorticity == true)
    {
      navier_stokes_operator->initialize_vector_velocity(vorticity);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::vector;
      sol.name        = "vorticity";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u();
      sol.vector      = &vorticity;
      this->additional_fields.push_back(sol);
    }

    // divergence
    if(output_data.write_divergence == true)
    {
      navier_stokes_operator->initialize_vector_velocity_scalar(divergence);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::scalar;
      sol.name        = "div_u";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
      sol.vector      = &divergence;
      this->additional_fields.push_back(sol);
    }

    // velocity magnitude
    if(output_data.write_velocity_magnitude == true)
    {
      navier_stokes_operator->initialize_vector_velocity_scalar(velocity_magnitude);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::scalar;
      sol.name        = "velocity_magnitude";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
      sol.vector      = &velocity_magnitude;
      this->additional_fields.push_back(sol);
    }

    // vorticity magnitude
    if(output_data.write_vorticity_magnitude == true)
    {
      navier_stokes_operator->initialize_vector_velocity_scalar(vorticity_magnitude);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::scalar;
      sol.name        = "vorticity_magnitude";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
      sol.vector      = &vorticity_magnitude;
      this->additional_fields.push_back(sol);
    }


    // streamfunction
    if(output_data.write_streamfunction == true)
    {
      navier_stokes_operator->initialize_vector_velocity_scalar(streamfunction);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::scalar;
      sol.name        = "streamfunction";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
      sol.vector      = &streamfunction;
      this->additional_fields.push_back(sol);
    }

    // q criterion
    if(output_data.write_q_criterion == true)
    {
      navier_stokes_operator->initialize_vector_velocity_scalar(q_criterion);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::scalar;
      sol.name        = "q_criterion";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
      sol.vector      = &q_criterion;
      this->additional_fields.push_back(sol);
    }

    // processor id
    if(output_data.write_processor_id == true)
    {
      navier_stokes_operator->initialize_vector_velocity_scalar(processor_id);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::scalar;
      sol.name        = "processor_id";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
      sol.vector      = &processor_id;
      this->additional_fields.push_back(sol);
    }

    // mean velocity
    if(output_data.mean_velocity.calculate == true)
    {
      navier_stokes_operator->initialize_vector_velocity(mean_velocity);

      SolutionField<dim, Number> sol;
      sol.type        = SolutionFieldType::vector;
      sol.name        = "mean_velocity";
      sol.dof_handler = &navier_stokes_operator->get_dof_handler_u();
      sol.vector      = &mean_velocity;
      this->additional_fields.push_back(sol);
    }
  }
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::compute_processor_id(VectorType & dst) const
{
  dst = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
}

template<int dim, typename Number>
void
OutputGenerator<dim, Number>::compute_mean_velocity(VectorType &       mean_velocity,
                                                    VectorType const & velocity,
                                                    double const       time,
                                                    int const          time_step_number)
{
  if(time >= output_data.mean_velocity.sample_start_time &&
     time <= output_data.mean_velocity.sample_end_time &&
     time_step_number % output_data.mean_velocity.sample_every_timesteps == 0)
  {
    mean_velocity.sadd((double)counter_mean_velocity, 1.0, velocity);
    ++counter_mean_velocity;
    mean_velocity *= 1. / (double)counter_mean_velocity;
  }
}


template<int dim, typename Number>
void
OutputGenerator<dim, Number>::calculate_additional_fields(VectorType const & velocity,
                                                          double const &     time,
                                                          int const &        time_step_number)
{
  if(output_data.write_output)
  {
    bool vorticity_is_up_to_date = false;
    if(output_data.write_vorticity == true)
    {
      navier_stokes_operator->compute_vorticity(vorticity, velocity);
      vorticity_is_up_to_date = true;
    }
    if(output_data.write_divergence == true)
    {
      navier_stokes_operator->compute_divergence(divergence, velocity);
    }
    if(output_data.write_velocity_magnitude == true)
    {
      navier_stokes_operator->compute_velocity_magnitude(velocity_magnitude, velocity);
    }
    if(output_data.write_vorticity_magnitude == true)
    {
      AssertThrow(vorticity_is_up_to_date == true,
                  ExcMessage("Vorticity vector needs to be updated to compute its magnitude."));

      navier_stokes_operator->compute_vorticity_magnitude(vorticity_magnitude, vorticity);
    }
    if(output_data.write_streamfunction == true)
    {
      AssertThrow(vorticity_is_up_to_date == true,
                  ExcMessage("Vorticity vector needs to be updated to compute its magnitude."));

      navier_stokes_operator->compute_streamfunction(streamfunction, vorticity);
    }
    if(output_data.write_q_criterion == true)
    {
      navier_stokes_operator->compute_q_criterion(q_criterion, velocity);
    }
    if(output_data.write_processor_id == true)
    {
      compute_processor_id(processor_id);
    }
    if(output_data.mean_velocity.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problems
        compute_mean_velocity(mean_velocity, velocity, time, time_step_number);
      else // time_step_number < 0 -> steady problems
        AssertThrow(false, ExcMessage("Mean velocity can only be computed for unsteady problems."));
    }
  }
}

template class OutputGenerator<2, float>;
template class OutputGenerator<2, double>;

template class OutputGenerator<3, float>;
template class OutputGenerator<3, double>;

} // namespace IncNS
