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
PostProcessor<dim, Number>::setup(Operator const & pde_operator)
{
  output_generator.setup(pde_operator,
                         pde_operator.get_dof_handler_u(),
                         pde_operator.get_dof_handler_p(),
                         pde_operator.get_mapping(),
                         pp_data.output_data);

  error_calculator_u.setup(pde_operator.get_dof_handler_u(),
                           pde_operator.get_mapping(),
                           pp_data.error_data_u);

  error_calculator_p.setup(pde_operator.get_dof_handler_p(),
                           pde_operator.get_mapping(),
                           pp_data.error_data_p);

  lift_and_drag_calculator.setup(pde_operator.get_dof_handler_u(),
                                 pde_operator.get_matrix_free(),
                                 pde_operator.get_dof_index_velocity(),
                                 pde_operator.get_dof_index_pressure(),
                                 pde_operator.get_quad_index_velocity_linear(),
                                 pp_data.lift_and_drag_data);

  pressure_difference_calculator.setup(pde_operator.get_dof_handler_p(),
                                       pde_operator.get_mapping(),
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
                                           pde_operator.get_dof_handler_u().get_triangulation(),
                                           pp_data.kinetic_energy_spectrum_data);

  line_plot_calculator.setup(pde_operator.get_dof_handler_u(),
                             pde_operator.get_dof_handler_p(),
                             pde_operator.get_mapping(),
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
