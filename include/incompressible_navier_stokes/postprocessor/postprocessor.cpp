/*
 * postprocessor.cpp
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

#include "postprocessor.h"

#include "../spatial_discretization/dg_navier_stokes_base.h"

namespace IncNS
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & postprocessor_data)
  : pp_data(postprocessor_data)
{
}

template<int dim, typename Number>
PostProcessor<dim, Number>::~PostProcessor()
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(NavierStokesOperator const &    navier_stokes_operator_in,
                                  DoFHandler<dim> const &         dof_handler_velocity_in,
                                  DoFHandler<dim> const &         dof_handler_pressure_in,
                                  Mapping<dim> const &            mapping_in,
                                  MatrixFree<dim, Number> const & matrix_free_in)
{
  output_generator.setup(navier_stokes_operator_in,
                         dof_handler_velocity_in,
                         dof_handler_pressure_in,
                         mapping_in,
                         pp_data.output_data);

  error_calculator_u.setup(dof_handler_velocity_in, mapping_in, pp_data.error_data_u);

  error_calculator_p.setup(dof_handler_pressure_in, mapping_in, pp_data.error_data_p);

  lift_and_drag_calculator.setup(dof_handler_velocity_in,
                                 matrix_free_in,
                                 navier_stokes_operator_in.get_dof_index_velocity(),
                                 navier_stokes_operator_in.get_dof_index_pressure(),
                                 navier_stokes_operator_in.get_quad_index_velocity_linear(),
                                 pp_data.lift_and_drag_data);

  pressure_difference_calculator.setup(dof_handler_pressure_in,
                                       mapping_in,
                                       pp_data.pressure_difference_data);

  div_and_mass_error_calculator.setup(matrix_free_in,
                                      navier_stokes_operator_in.get_dof_index_velocity(),
                                      navier_stokes_operator_in.get_quad_index_velocity_linear(),
                                      pp_data.mass_data);

  kinetic_energy_calculator.setup(navier_stokes_operator_in,
                                  matrix_free_in,
                                  navier_stokes_operator_in.get_dof_index_velocity(),
                                  navier_stokes_operator_in.get_quad_index_velocity_linear(),
                                  pp_data.kinetic_energy_data);

  kinetic_energy_spectrum_calculator.setup(matrix_free_in,
                                           dof_handler_velocity_in.get_triangulation(),
                                           pp_data.kinetic_energy_spectrum_data);

  line_plot_calculator.setup(dof_handler_velocity_in,
                             dof_handler_pressure_in,
                             mapping_in,
                             pp_data.line_plot_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const & velocity,
                                              VectorType const & pressure,
                                              double const       time,
                                              int const          time_step_number)
{
  /*
   *  write output
   */
  output_generator.evaluate(velocity, pressure, time, time_step_number);

  /*
   *  calculate error
   */
  error_calculator_u.evaluate(velocity, time, time_step_number);
  error_calculator_p.evaluate(pressure, time, time_step_number);

  /*
   *  calculation of lift and drag coefficients
   */
  lift_and_drag_calculator.evaluate(velocity, pressure, time);

  /*
   *  calculation of pressure difference
   */
  pressure_difference_calculator.evaluate(pressure, time);

  /*
   *  Analysis of divergence and mass error
   */
  div_and_mass_error_calculator.evaluate(velocity, time, time_step_number);

  /*
   *  calculation of kinetic energy
   */
  kinetic_energy_calculator.evaluate(velocity, time, time_step_number);

  /*
   *  calculation of kinetic energy spectrum
   */
  kinetic_energy_spectrum_calculator.evaluate(velocity, time, time_step_number);

  /*
   *  Evaluate fields along lines
   */
  line_plot_calculator.evaluate(velocity, pressure);
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace IncNS
