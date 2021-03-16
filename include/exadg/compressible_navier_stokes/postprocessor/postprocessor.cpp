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

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & postprocessor_data,
                                          MPI_Comm const &               comm)
  : mpi_comm(comm),
    pp_data(postprocessor_data),
    output_generator(comm),
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
PostProcessor<dim, Number>::do_postprocessing(VectorType const & solution,
                                              double const       time,
                                              int const          time_step_number)
{
  /*
   * calculate derived quantities such as velocity, pressure, etc.
   */
  calculate_additional_vectors(solution);

  /*
   *  write output
   */
  output_generator.evaluate(solution, additional_fields, time, time_step_number);

  /*
   *  calculate error
   */
  error_calculator.evaluate(solution, time, time_step_number);

  /*
   *  calculation of lift and drag coefficients
   */
  lift_and_drag_calculator.evaluate(velocity, pressure, time);

  /*
   *  calculation of pressure difference
   */
  pressure_difference_calculator.evaluate(pressure, time);

  /*
   *  calculation of kinetic energy
   */
  kinetic_energy_calculator.evaluate(velocity, time, time_step_number);

  /*
   *  calculation of kinetic energy spectrum
   */
  kinetic_energy_spectrum_calculator.evaluate(velocity, time, time_step_number);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_additional_vectors()
{
  if(pp_data.output_data.write_pressure == true)
  {
    navier_stokes_operator->initialize_dof_vector_scalar(pressure);

    SolutionField<dim, Number> field;
    field.type        = SolutionFieldType::scalar;
    field.name        = "pressure";
    field.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
    field.vector      = &pressure;
    additional_fields.push_back(field);
  }

  // velocity
  if(pp_data.output_data.write_velocity == true)
  {
    navier_stokes_operator->initialize_dof_vector_dim_components(velocity);

    SolutionField<dim, Number> field;
    field.type        = SolutionFieldType::vector;
    field.name        = "velocity";
    field.dof_handler = &navier_stokes_operator->get_dof_handler_vector();
    field.vector      = &velocity;
    additional_fields.push_back(field);
  }

  // temperature
  if(pp_data.output_data.write_temperature == true)
  {
    navier_stokes_operator->initialize_dof_vector_scalar(temperature);

    SolutionField<dim, Number> field;
    field.type        = SolutionFieldType::scalar;
    field.name        = "temperature";
    field.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
    field.vector      = &temperature;
    additional_fields.push_back(field);
  }

  // vorticity
  if(pp_data.output_data.write_vorticity == true)
  {
    navier_stokes_operator->initialize_dof_vector_dim_components(vorticity);

    SolutionField<dim, Number> field;
    field.type        = SolutionFieldType::vector;
    field.name        = "vorticity";
    field.dof_handler = &navier_stokes_operator->get_dof_handler_vector();
    field.vector      = &vorticity;
    additional_fields.push_back(field);
  }

  // divergence
  if(pp_data.output_data.write_divergence == true)
  {
    navier_stokes_operator->initialize_dof_vector_scalar(divergence);

    SolutionField<dim, Number> field;
    field.type        = SolutionFieldType::scalar;
    field.name        = "velocity_divergence";
    field.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
    field.vector      = &divergence;
    additional_fields.push_back(field);
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::calculate_additional_vectors(VectorType const & solution)
{
  if((pp_data.output_data.write_output == true && pp_data.output_data.write_pressure == true) ||
     pp_data.calculate_pressure == true)
  {
    navier_stokes_operator->compute_pressure(pressure, solution);
  }

  if((pp_data.output_data.write_output == true && pp_data.output_data.write_velocity == true) ||
     (pp_data.output_data.write_output == true && pp_data.output_data.write_vorticity == true) ||
     (pp_data.output_data.write_output == true && pp_data.output_data.write_divergence == true) ||
     pp_data.calculate_velocity == true)
  {
    navier_stokes_operator->compute_velocity(velocity, solution);
  }

  if(pp_data.output_data.write_output == true && pp_data.output_data.write_temperature == true)
  {
    navier_stokes_operator->compute_temperature(temperature, solution);
  }

  if(pp_data.output_data.write_output == true && pp_data.output_data.write_vorticity == true)
  {
    navier_stokes_operator->compute_vorticity(vorticity, velocity);
  }

  if(pp_data.output_data.write_output == true && pp_data.output_data.write_divergence == true)
  {
    navier_stokes_operator->compute_divergence(divergence, velocity);
  }
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace CompNS
} // namespace ExaDG
