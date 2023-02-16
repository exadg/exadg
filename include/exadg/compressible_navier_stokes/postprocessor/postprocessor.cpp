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

  initialize_derived_fields();

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
  invalidate_derived_fields();

  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> additional_fields_vtu;

    if(pp_data.output_data.write_pressure)
    {
      pressure.evaluate(solution);
      additional_fields_vtu.push_back(&pressure);
    }
    if(pp_data.output_data.write_velocity)
    {
      velocity.evaluate(solution);
      additional_fields_vtu.push_back(&velocity);
    }
    if(pp_data.output_data.write_vorticity)
    {
      vorticity.evaluate(velocity.evaluate_get(solution));
      additional_fields_vtu.push_back(&vorticity);
    }
    if(pp_data.output_data.write_divergence)
    {
      divergence.evaluate(velocity.evaluate_get(solution));
      additional_fields_vtu.push_back(&divergence);
    }
    if(pp_data.output_data.write_shear_rate)
    {
      shear_rate.evaluate(velocity.evaluate_get(solution));
      additional_fields_vtu.push_back(&shear_rate);
    }
    if(pp_data.output_data.write_temperature)
    {
      temperature.evaluate(solution);
      additional_fields_vtu.push_back(&temperature);
    }

    output_generator.evaluate(solution,
                              additional_fields_vtu,
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
    lift_and_drag_calculator.evaluate(velocity.evaluate_get(solution),
                                      pressure.evaluate_get(solution),
                                      time);

  /*
   *  calculation of pressure difference
   */
  if(pressure_difference_calculator.time_control.needs_evaluation(time, time_step_number))
    pressure_difference_calculator.evaluate(pressure.evaluate_get(solution), time);

  /*
   *  calculation of kinetic energy
   */
  if(kinetic_energy_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    kinetic_energy_calculator.evaluate(velocity.evaluate_get(solution),
                                       time,
                                       Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculation of kinetic energy spectrum
   */
  if(kinetic_energy_spectrum_calculator.time_control.needs_evaluation(time, time_step_number))
  {
    kinetic_energy_spectrum_calculator.evaluate(velocity.evaluate_get(solution),
                                                time,
                                                Utilities::is_unsteady_timestep(time_step_number));
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_derived_fields()
{
  if(pp_data.output_data.write_pressure || pp_data.lift_and_drag_data.time_control_data.is_active ||
     pp_data.pressure_difference_data.time_control_data.is_active)
  {
    pressure.type              = SolutionFieldType::scalar;
    pressure.name              = "pressure";
    pressure.dof_handler       = &navier_stokes_operator->get_dof_handler_scalar();
    pressure.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_dof_vector_scalar(dst);
    };
    pressure.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_pressure(dst, src);
    };

    pressure.reinit();
  }

  // velocity
  if(pp_data.output_data.write_velocity || pp_data.output_data.write_vorticity ||
     pp_data.output_data.write_divergence ||
	 pp_data.output_data.write_shear_rate ||
	 pp_data.lift_and_drag_data.time_control_data.is_active ||
     pp_data.kinetic_energy_data.time_control_data.is_active ||
     pp_data.kinetic_energy_spectrum_data.time_control_data.is_active)
  {
    velocity.type              = SolutionFieldType::vector;
    velocity.name              = "velocity";
    velocity.dof_handler       = &navier_stokes_operator->get_dof_handler_vector();
    velocity.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_dof_vector_dim_components(dst);
    };
    velocity.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_velocity(dst, src);
    };

    velocity.reinit();
  }

  // vorticity
  if(pp_data.output_data.write_vorticity)
  {
    AssertThrow(pp_data.output_data.write_velocity == true,
                dealii::ExcMessage("You need to activate write_velocity."));

    vorticity.type              = SolutionFieldType::vector;
    vorticity.name              = "vorticity";
    vorticity.dof_handler       = &navier_stokes_operator->get_dof_handler_vector();
    vorticity.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_dof_vector_dim_components(dst);
    };
    vorticity.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_vorticity(dst, src);
    };

    vorticity.reinit();
  }

  // divergence
  if(pp_data.output_data.write_divergence)
  {
    AssertThrow(pp_data.output_data.write_velocity == true,
                dealii::ExcMessage("You need to activate write_velocity."));

    divergence.type              = SolutionFieldType::scalar;
    divergence.name              = "velocity_divergence";
    divergence.dof_handler       = &navier_stokes_operator->get_dof_handler_scalar();
    divergence.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_dof_vector_scalar(dst);
    };
    divergence.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_divergence(dst, src);
    };

    divergence.reinit();
  }

  // shear rate
  if(pp_data.output_data.write_shear_rate)
  {
    AssertThrow(pp_data.output_data.write_velocity == true,
                dealii::ExcMessage("You need to activate write_velocity."));

    shear_rate.type              = SolutionFieldType::scalar;
    shear_rate.name              = "shear_rate";
    shear_rate.dof_handler       = &navier_stokes_operator->get_dof_handler_scalar();
    shear_rate.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_dof_vector_scalar(dst);
    };
    shear_rate.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_shear_rate(dst, src);
    };

    shear_rate.reinit();
  }

  // temperature
  if(pp_data.output_data.write_temperature)
  {
    temperature.type              = SolutionFieldType::scalar;
    temperature.name              = "temperature";
    temperature.dof_handler       = &navier_stokes_operator->get_dof_handler_scalar();
    temperature.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_dof_vector_scalar(dst);
    };
    temperature.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_temperature(dst, src);
    };

    temperature.reinit();
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::invalidate_derived_fields()
{
  pressure.invalidate();
  velocity.invalidate();
  temperature.invalidate();
  vorticity.invalidate();
  divergence.invalidate();
  shear_rate.invalidate();
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace CompNS
} // namespace ExaDG
