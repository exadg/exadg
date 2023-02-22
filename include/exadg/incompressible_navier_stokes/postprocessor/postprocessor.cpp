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

  initialize_derived_fields();

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
  invalidate_derived_fields();

  /*
   *  compute mean velocity
   */
  if(time_control_mean_velocity.needs_evaluation(time, time_step_number))
  {
    AssertThrow(Utilities::is_unsteady_timestep(time_step_number),
                dealii::ExcMessage(
                  "Calculating mean velocity does not make sense for steady problems."));

    mean_velocity.evaluate(velocity);
  }


  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> additional_fields_vtu;
    std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> surface_fields_vtu;
    if(pp_data.output_data.write_vorticity)
    {
      vorticity.evaluate(velocity);
      additional_fields_vtu.push_back(&vorticity);
    }
    if(pp_data.output_data.write_vorticity_magnitude)
    {
      vorticity_magnitude.evaluate(vorticity.evaluate_get(velocity));
      additional_fields_vtu.push_back(&vorticity_magnitude);
    }
    if(pp_data.output_data.write_streamfunction)
    {
      streamfunction.evaluate(vorticity.evaluate_get(velocity));
      additional_fields_vtu.push_back(&streamfunction);
    }
    if(pp_data.output_data.write_divergence)
    {
      divergence.evaluate(velocity);
      additional_fields_vtu.push_back(&divergence);
    }
    if(pp_data.output_data.write_shear_rate)
    {
      shear_rate.evaluate(velocity);
      additional_fields_vtu.push_back(&shear_rate);
    }
    if(pp_data.output_data.write_velocity_magnitude)
    {
      velocity_magnitude.evaluate(velocity);
      additional_fields_vtu.push_back(&velocity_magnitude);
    }
    if(pp_data.output_data.write_wall_shear_stress_on_IDs.size() > 0)
    {
      wall_shear_stress.evaluate(velocity);
      if(dim == 3)
        surface_fields_vtu.push_back(&wall_shear_stress);
      else
        additional_fields_vtu.push_back(&wall_shear_stress);
    }
    if(pp_data.output_data.write_q_criterion)
    {
      q_criterion.evaluate(velocity);
      additional_fields_vtu.push_back(&q_criterion);
    }
    if(pp_data.output_data.mean_velocity.is_active)
    {
      additional_fields_vtu.push_back(&mean_velocity);
    }
    if(pp_data.output_data.write_cfl)
    {
      cfl_vector.evaluate(velocity);
      additional_fields_vtu.push_back(&cfl_vector);
    }

    output_generator.evaluate(velocity,
                              pressure,
                              additional_fields_vtu,
                              surface_fields_vtu,
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
PostProcessor<dim, Number>::initialize_derived_fields()
{
  // vorticity
  if(pp_data.output_data.write_vorticity || pp_data.output_data.write_streamfunction ||
     pp_data.output_data.write_vorticity_magnitude)
  {
    vorticity.type              = SolutionFieldType::vector;
    vorticity.name              = "vorticity";
    vorticity.dof_handler       = &navier_stokes_operator->get_dof_handler_u();
    vorticity.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity(dst);
    };
    vorticity.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_vorticity(dst, src);
    };

    vorticity.reinit();
  }

  // vorticity magnitude
  if(pp_data.output_data.write_vorticity_magnitude)
  {
    AssertThrow(pp_data.output_data.write_vorticity == true,
                dealii::ExcMessage("You need to activate write_vorticity."));

    vorticity_magnitude.type              = SolutionFieldType::scalar;
    vorticity_magnitude.name              = "vorticity_magnitude";
    vorticity_magnitude.dof_handler       = &navier_stokes_operator->get_dof_handler_u_scalar();
    vorticity_magnitude.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity_scalar(dst);
    };
    vorticity_magnitude.recompute_solution_field = [&](VectorType & dst, const VectorType & src) {
      navier_stokes_operator->compute_vorticity_magnitude(dst, src);
    };

    vorticity_magnitude.reinit();
  }

  // wall shear stress
  if(pp_data.output_data.write_wall_shear_stress_on_IDs.size() > 0)
  {
    wall_shear_stress.type              = SolutionFieldType::vector;
    wall_shear_stress.name              = "wall_shear_stress";
    wall_shear_stress.dof_handler       = &navier_stokes_operator->get_dof_handler_u();
    wall_shear_stress.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity(dst);
    };
    wall_shear_stress.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_wall_shear_stress(
        dst, src, pp_data.output_data.write_wall_shear_stress_on_IDs);
    };

    wall_shear_stress.reinit();
  }

  // streamfunction
  if(pp_data.output_data.write_streamfunction)
  {
    AssertThrow(pp_data.output_data.write_vorticity == true,
                dealii::ExcMessage("You need to activate write_vorticity."));

    streamfunction.type              = SolutionFieldType::scalar;
    streamfunction.name              = "streamfunction";
    streamfunction.dof_handler       = &navier_stokes_operator->get_dof_handler_u_scalar();
    streamfunction.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity_scalar(dst);
    };
    streamfunction.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_streamfunction(dst, src);
    };

    streamfunction.reinit();
  }

  // divergence
  if(pp_data.output_data.write_divergence)
  {
    divergence.type              = SolutionFieldType::scalar;
    divergence.name              = "div_u";
    divergence.dof_handler       = &navier_stokes_operator->get_dof_handler_u_scalar();
    divergence.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity_scalar(dst);
    };
    divergence.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_divergence(dst, src);
    };

    divergence.reinit();
  }

  // shear rate
  if(pp_data.output_data.write_shear_rate)
  {
    shear_rate.type              = SolutionFieldType::scalar;
    shear_rate.name              = "shear_rate";
    shear_rate.dof_handler       = &navier_stokes_operator->get_dof_handler_u_scalar();
    shear_rate.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity_scalar(dst);
    };
    shear_rate.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_shear_rate(dst, src);
    };

    shear_rate.reinit();
  }

  // velocity magnitude
  if(pp_data.output_data.write_velocity_magnitude)
  {
    velocity_magnitude.type              = SolutionFieldType::scalar;
    velocity_magnitude.name              = "velocity_magnitude";
    velocity_magnitude.dof_handler       = &navier_stokes_operator->get_dof_handler_u_scalar();
    velocity_magnitude.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity_scalar(dst);
    };
    velocity_magnitude.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_velocity_magnitude(dst, src);
    };

    velocity_magnitude.reinit();
  }

  // q criterion
  if(pp_data.output_data.write_q_criterion)
  {
    q_criterion.type              = SolutionFieldType::scalar;
    q_criterion.name              = "q_criterion";
    q_criterion.dof_handler       = &navier_stokes_operator->get_dof_handler_u_scalar();
    q_criterion.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity_scalar(dst);
    };
    q_criterion.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      navier_stokes_operator->compute_q_criterion(dst, src);
    };

    q_criterion.reinit();
  }

  // mean velocity
  if(pp_data.output_data.mean_velocity.is_active)
  {
    mean_velocity.type              = SolutionFieldType::vector;
    mean_velocity.name              = "mean_velocity";
    mean_velocity.dof_handler       = &navier_stokes_operator->get_dof_handler_u();
    mean_velocity.initialize_vector = [&](VectorType & dst) {
      navier_stokes_operator->initialize_vector_velocity(dst);
    };
    mean_velocity.recompute_solution_field = [&](VectorType & dst, VectorType const & velocity) {
      unsigned int const counter = time_control_mean_velocity.get_counter();
      dst.sadd((double)counter, 1.0, velocity);
      dst *= 1. / (double)(counter + 1);
    };

    mean_velocity.reinit();
  }

  // cfl
  if(pp_data.output_data.write_cfl)
  {
    cfl_vector.type                     = SolutionFieldType::cellwise;
    cfl_vector.name                     = "cfl_relative";
    cfl_vector.initialize_vector        = [&](VectorType &) {};
    cfl_vector.recompute_solution_field = [&](VectorType & dst, VectorType const & src) {
      // This time step size corresponds to CFL = 1.
      auto const time_step_size = navier_stokes_operator->calculate_time_step_cfl(src);
      // The computed cell-vector of CFL values contains relative CFL numbers with a value of
      // CFL = 1 in the most critical cell and CFL < 1 in other cells.
      navier_stokes_operator->calculate_cfl_from_time_step(dst, src, time_step_size);
    };

    cfl_vector.reinit();
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::invalidate_derived_fields()
{
  vorticity.invalidate();
  divergence.invalidate();
  shear_rate.invalidate();
  velocity_magnitude.invalidate();
  vorticity_magnitude.invalidate();
  wall_shear_stress.invalidate();
  streamfunction.invalidate();
  q_criterion.invalidate();
  cfl_vector.invalidate();
  mean_velocity.invalidate();
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace IncNS
} // namespace ExaDG
