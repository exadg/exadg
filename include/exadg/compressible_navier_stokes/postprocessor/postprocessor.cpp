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

#include <exadg/compressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/compressible_navier_stokes/spatial_discretization/operator.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & postprocessor_data,
                                          MPI_Comm const &               comm)
  : mpi_comm(comm),
    pp_data(postprocessor_data),
    output_generator(comm),
    pointwise_output_generator(comm),
    error_calculator(comm),
    lift_and_drag_calculator(comm),
    pressure_difference_calculator(comm),
    kinetic_energy_calculator(comm),
    kinetic_energy_spectrum_calculator(comm)
{
}

template<int dim, typename Number>
PostProcessor<dim, Number>::~PostProcessor()
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator<dim, Number> const & pde_operator)
{
  navier_stokes_operator = &pde_operator;

  initialize_additional_vectors();

  output_generator.setup(pde_operator.get_dof_handler(),
                         pde_operator.get_mapping(),
                         pp_data.output_data);

  pointwise_output_generator.setup(pde_operator.get_dof_handler(),
                                   pde_operator.get_mapping(),
                                   pp_data.pointwise_output_data);

  error_calculator.setup(pde_operator.get_dof_handler(),
                         pde_operator.get_mapping(),
                         pp_data.error_data);

  lift_and_drag_calculator.setup(pde_operator.get_dof_handler(),
                                 pde_operator.get_matrix_free(),
                                 pde_operator.get_dof_index_vector(),
                                 pde_operator.get_dof_index_scalar(),
                                 pde_operator.get_quad_index_standard(),
                                 pp_data.lift_and_drag_data);

  pressure_difference_calculator.setup(pde_operator.get_dof_handler_scalar(),
                                       pde_operator.get_mapping(),
                                       pp_data.pressure_difference_data);

  kinetic_energy_calculator.setup(pde_operator.get_matrix_free(),
                                  pde_operator.get_dof_index_vector(),
                                  pde_operator.get_quad_index_standard(),
                                  pp_data.kinetic_energy_data);

  kinetic_energy_spectrum_calculator.setup(pde_operator.get_matrix_free(),
                                           pde_operator.get_dof_handler(),
                                           pp_data.kinetic_energy_spectrum_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     solution,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  reinit_additional_fields(solution);

  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    output_generator.evaluate(solution,
                              evaluate_get(additional_fields_vtu,
                                           Utilities::is_unsteady_timestep(time_step_number)),
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  write pointwise output
   */
  if(pointwise_output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    pointwise_output_generator.evaluate(solution,
                                        time,
                                        Utilities::is_unsteady_timestep(time_step_number));
  }
  /*
   *  calculate error
   */
  if(error_calculator.time_control.needs_evaluation(time, time_step_number))
    error_calculator.evaluate(solution, time, Utilities::is_unsteady_timestep(time_step_number));

  /*
   *  calculation of lift and drag coefficients
   */
  if(lift_and_drag_calculator.time_control.needs_evaluation(time, time_step_number))
    lift_and_drag_calculator.evaluate(
      evaluate_get(velocity, Utilities::is_unsteady_timestep(time_step_number)),
      evaluate_get(pressure, Utilities::is_unsteady_timestep(time_step_number)),
      time);

  /*
   *  calculation of pressure difference
   */
  if(pressure_difference_calculator.time_control.needs_evaluation(time, time_step_number))
    pressure_difference_calculator.evaluate(
      evaluate_get(pressure, Utilities::is_unsteady_timestep(time_step_number)), time);

  /*
   *  calculation of kinetic energy
   */
  if(kinetic_energy_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    kinetic_energy_calculator.evaluate(
      evaluate_get(velocity, Utilities::is_unsteady_timestep(time_step_number)),
      time,
      Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculation of kinetic energy spectrum
   */
  if(kinetic_energy_spectrum_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    kinetic_energy_spectrum_calculator.evaluate(
      evaluate_get(velocity, Utilities::is_unsteady_timestep(time_step_number)),
      time,
      Utilities::is_unsteady_timestep(time_step_number));
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_additional_vectors()
{
  if(pp_data.output_data.write_pressure || pp_data.lift_and_drag_data.time_control_data.is_active ||
     pp_data.pressure_difference_data.time_control_data.is_active)
  {
    pressure.type        = SolutionFieldType::scalar;
    pressure.name        = "pressure";
    pressure.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
    navier_stokes_operator->initialize_dof_vector_scalar(pressure.get_vector_reference());
    pressure.recompute_solution_field = [&](VectorType & dst, VectorType const & src, bool const) {
      navier_stokes_operator->compute_pressure(dst, src);
    };

    if(pp_data.output_data.write_pressure)
      additional_fields_vtu.push_back(&pressure);
  }

  // velocity
  if(pp_data.output_data.write_velocity || pp_data.output_data.write_vorticity ||
     pp_data.output_data.write_divergence ||
     pp_data.lift_and_drag_data.time_control_data.is_active ||
     pp_data.kinetic_energy_data.time_control_data.is_active ||
     pp_data.kinetic_energy_spectrum_data.time_control_data.is_active)
  {
    velocity.type        = SolutionFieldType::vector;
    velocity.name        = "velocity";
    velocity.dof_handler = &navier_stokes_operator->get_dof_handler_vector();
    navier_stokes_operator->initialize_dof_vector_dim_components(velocity.get_vector_reference());
    velocity.recompute_solution_field = [&](VectorType & dst, VectorType const & src, bool const) {
      navier_stokes_operator->compute_velocity(dst, src);
    };

    if(pp_data.output_data.write_velocity)
      additional_fields_vtu.push_back(&velocity);
  }

  // temperature
  if(pp_data.output_data.write_temperature)
  {
    temperature.type        = SolutionFieldType::scalar;
    temperature.name        = "temperature";
    temperature.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
    navier_stokes_operator->initialize_dof_vector_scalar(temperature.get_vector_reference());
    temperature.recompute_solution_field =
      [&](VectorType & dst, VectorType const & src, bool const) {
        navier_stokes_operator->compute_temperature(dst, src);
      };

    additional_fields_vtu.push_back(&temperature);
  }

  // vorticity
  if(pp_data.output_data.write_vorticity)
  {
    vorticity.type        = SolutionFieldType::vector;
    vorticity.name        = "vorticity";
    vorticity.dof_handler = &navier_stokes_operator->get_dof_handler_vector();
    navier_stokes_operator->initialize_dof_vector_dim_components(vorticity.get_vector_reference());
    vorticity.recompute_solution_field = [&](VectorType & dst, VectorType const &, bool const) {
      navier_stokes_operator->compute_vorticity(dst, velocity.get_vector());
    };

    additional_fields_vtu.push_back(&vorticity);
  }

  // divergence
  if(pp_data.output_data.write_divergence)
  {
    divergence.type        = SolutionFieldType::scalar;
    divergence.name        = "velocity_divergence";
    divergence.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
    navier_stokes_operator->initialize_dof_vector_scalar(divergence.get_vector_reference());
    divergence.recompute_solution_field = [&](VectorType & dst, VectorType const &, bool const) {
      navier_stokes_operator->compute_divergence(dst, velocity.get_vector());
    };

    additional_fields_vtu.push_back(&divergence);
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::reinit_additional_fields(VectorType const & solution)
{
  pressure.reinit(solution);
  velocity.reinit(solution);
  temperature.reinit(solution);
  vorticity.reinit(solution);
  divergence.reinit(solution);
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace CompNS
} // namespace ExaDG
