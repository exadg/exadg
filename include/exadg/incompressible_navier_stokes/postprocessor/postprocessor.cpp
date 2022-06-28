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

#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>

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
  output_generator.setup(pde_operator,
                         pde_operator.get_dof_handler_u(),
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

  base_operator = &pde_operator;
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const & velocity,
                                              VectorType const & pressure,
                                              double const       time,
                                              int const          time_step_number)
{
  // We need to distribute the dofs for HDIV before computing the error since
  // dealii::VectorTools::integrate_difference() does not take the periodic boundary
  // constraints into account like MatrixFree does, hence reading the wrong value.
  // distribute() updates the constrained values.
  if(base_operator->get_spatial_discretization() == SpatialDiscretization::HDIV)
    base_operator->get_constraint_u().distribute(const_cast<VectorType &>(velocity));

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
} // namespace ExaDG
