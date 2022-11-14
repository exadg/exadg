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

#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & postprocessor_data,
                                          MPI_Comm const &               comm)
  : mpi_comm(comm),
    pp_data(postprocessor_data),
    output_generator(comm),
    error_calculator_u(comm),
    error_calculator_p(comm),
    lift_and_drag_calculator(comm),
    pressure_difference_calculator(comm),
    div_and_mass_error_calculator(comm),
    kinetic_energy_calculator(comm),
    kinetic_energy_spectrum_calculator(comm),
    line_plot_calculator(comm)
{
}

template<int dim, typename Number>
PostProcessor<dim, Number>::~PostProcessor()
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator const & pde_operator)
{
  navier_stokes_operator = &pde_operator;

  time_control_mean_velocity.setup(pp_data.output_data.mean_velocity);

  initialize_additional_vectors();

  output_generator.setup(pde_operator.get_dof_handler_u(),
                         pde_operator.get_dof_handler_p(),
                         *pde_operator.get_mapping(),
                         pp_data.output_data);

  error_calculator_u.setup(pde_operator.get_dof_handler_u(),
                           *pde_operator.get_mapping(),
                           pp_data.error_data_u);

  error_calculator_p.setup(pde_operator.get_dof_handler_p(),
                           *pde_operator.get_mapping(),
                           pp_data.error_data_p);

  lift_and_drag_calculator.setup(pde_operator.get_dof_handler_u(),
                                 pde_operator.get_matrix_free(),
                                 pde_operator.get_dof_index_velocity(),
                                 pde_operator.get_dof_index_pressure(),
                                 pde_operator.get_quad_index_velocity_linear(),
                                 pp_data.lift_and_drag_data);

  pressure_difference_calculator.setup(pde_operator.get_dof_handler_p(),
                                       *pde_operator.get_mapping(),
                                       pp_data.pressure_difference_data);

  div_and_mass_error_calculator.setup(pde_operator.get_matrix_free(),
                                      pde_operator.get_dof_index_velocity(),
                                      pde_operator.get_quad_index_velocity_linear(),
                                      pp_data.mass_data);

  kinetic_energy_calculator.setup(pde_operator,
                                  pde_operator.get_matrix_free(),
                                  pde_operator.get_dof_index_velocity(),
                                  pde_operator.get_quad_index_velocity_linear(),
                                  pp_data.kinetic_energy_data);

  kinetic_energy_spectrum_calculator.setup(pde_operator.get_matrix_free(),
                                           pde_operator.get_dof_handler_u(),
                                           pp_data.kinetic_energy_spectrum_data);

  line_plot_calculator.setup(pde_operator.get_dof_handler_u(),
                             pde_operator.get_dof_handler_p(),
                             *pde_operator.get_mapping(),
                             pp_data.line_plot_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     velocity,
                                              VectorType const &     pressure,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  // TODO: NOTE THAT CALLING THE FUNCTION HERE RESULTS IN SLOWER CODE. THIS IS REPORTED AN RESOLVED
  // IN PR #245. SO WE SHOULD ONLY MERGE THIS PR IF ALSO #245 IS READY
  /*
   * calculate derived quantities such as vorticity, divergence, etc.
   */
  calculate_additional_vectors(velocity, time, time_step_number);

  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    output_generator.evaluate(velocity,
                              pressure,
                              additional_fields,
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculate error
   */
  if(error_calculator_u.time_control.needs_evaluation(time, time_step_number))
    error_calculator_u.evaluate(velocity, time, Utilities::is_unsteady_timestep(time_step_number));
  if(error_calculator_p.time_control.needs_evaluation(time, time_step_number))
    error_calculator_p.evaluate(pressure, time, Utilities::is_unsteady_timestep(time_step_number));

  /*
   *  calculation of lift and drag coefficients
   */
  if(lift_and_drag_calculator.time_control.needs_evaluation(time, time_step_number))
    lift_and_drag_calculator.evaluate(velocity, pressure, time);

  /*
   *  calculation of pressure difference
   */
  if(pressure_difference_calculator.time_control.needs_evaluation(time, time_step_number))
    pressure_difference_calculator.evaluate(pressure, time);

  /*
   *  Analysis of divergence and mass error
   */
  if(div_and_mass_error_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    div_and_mass_error_calculator.evaluate(velocity,
                                           time,
                                           Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculation of kinetic energy
   */
  if(kinetic_energy_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    kinetic_energy_calculator.evaluate(velocity,
                                       time,
                                       Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculation of kinetic energy spectrum
   */
  if(kinetic_energy_spectrum_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    kinetic_energy_spectrum_calculator.evaluate(velocity,
                                                time,
                                                Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  Evaluate fields along lines
   */
  if(line_plot_calculator.time_control.needs_evaluation(time, time_step_number))
    line_plot_calculator.evaluate(velocity, pressure);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_additional_vectors()
{
  // vorticity
  if(pp_data.output_data.write_vorticity || pp_data.output_data.write_streamfunction ||
     pp_data.output_data.write_vorticity_magnitude)
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
  if(pp_data.output_data.write_divergence == true)
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
  if(pp_data.output_data.write_velocity_magnitude == true)
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
  if(pp_data.output_data.write_vorticity_magnitude == true)
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
  if(pp_data.output_data.write_streamfunction == true)
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
  if(pp_data.output_data.write_q_criterion == true)
  {
    navier_stokes_operator->initialize_vector_velocity_scalar(q_criterion);

    SolutionField<dim, Number> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "q_criterion";
    sol.dof_handler = &navier_stokes_operator->get_dof_handler_u_scalar();
    sol.vector      = &q_criterion;
    this->additional_fields.push_back(sol);
  }

  // mean velocity
  if(pp_data.output_data.mean_velocity.is_active == true)
  {
    navier_stokes_operator->initialize_vector_velocity(mean_velocity);

    SolutionField<dim, Number> sol;
    sol.type        = SolutionFieldType::vector;
    sol.name        = "mean_velocity";
    sol.dof_handler = &navier_stokes_operator->get_dof_handler_u();
    sol.vector      = &mean_velocity;
    this->additional_fields.push_back(sol);
  }

  // cfl
  if(pp_data.output_data.write_cfl)
  {
    SolutionField<dim, Number> sol;
    sol.type   = SolutionFieldType::cellwise;
    sol.name   = "cfl_relative";
    sol.vector = &cfl_vector;
    this->additional_fields.push_back(sol);
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::compute_mean_velocity(VectorType &       mean_velocity,
                                                  VectorType const & velocity,
                                                  bool const         unsteady)
{
  AssertThrow(unsteady,
              dealii::ExcMessage(
                "Calculating mean velocity does not make sense for steady problems."));

  unsigned int const counter = time_control_mean_velocity.get_counter();
  mean_velocity.sadd((double)counter, 1.0, velocity);
  mean_velocity *= 1. / (double)(counter + 1);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::calculate_additional_vectors(VectorType const &     velocity,
                                                         double const           time,
                                                         types::time_step const time_step_number)
{
  bool vorticity_is_up_to_date = false;
  if(pp_data.output_data.write_vorticity || pp_data.output_data.write_streamfunction ||
     pp_data.output_data.write_vorticity_magnitude)
  {
    navier_stokes_operator->compute_vorticity(vorticity, velocity);
    vorticity_is_up_to_date = true;
  }

  if(pp_data.output_data.write_divergence == true)
  {
    navier_stokes_operator->compute_divergence(divergence, velocity);
  }

  if(pp_data.output_data.write_velocity_magnitude == true)
  {
    navier_stokes_operator->compute_velocity_magnitude(velocity_magnitude, velocity);
  }

  if(pp_data.output_data.write_vorticity_magnitude == true)
  {
    AssertThrow(vorticity_is_up_to_date == true,
                dealii::ExcMessage(
                  "Vorticity vector needs to be updated to compute its magnitude."));

    navier_stokes_operator->compute_vorticity_magnitude(vorticity_magnitude, vorticity);
  }

  if(pp_data.output_data.write_streamfunction == true)
  {
    AssertThrow(vorticity_is_up_to_date == true,
                dealii::ExcMessage(
                  "Vorticity vector needs to be updated to compute its magnitude."));

    navier_stokes_operator->compute_streamfunction(streamfunction, vorticity);
  }

  if(pp_data.output_data.write_q_criterion == true)
  {
    navier_stokes_operator->compute_q_criterion(q_criterion, velocity);
  }

  if(pp_data.output_data.mean_velocity.is_active == true)
  {
    if(time_control_mean_velocity.needs_evaluation(time, time_step_number))
    {
      compute_mean_velocity(mean_velocity,
                            velocity,
                            Utilities::is_unsteady_timestep(time_step_number));
    }
  }

  if(pp_data.output_data.write_cfl)
  {
    // This time step size corresponds to CFL = 1.
    auto const time_step_size = navier_stokes_operator->calculate_time_step_cfl(velocity);

    // The computed cell-vector of CFL values contains relative CFL numbers with a value of
    // CFL = 1 in the most critical cell and CFL < 1 in other cells.
    navier_stokes_operator->calculate_cfl_from_time_step(cfl_vector, velocity, time_step_size);
  }
}


template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace IncNS
} // namespace ExaDG
