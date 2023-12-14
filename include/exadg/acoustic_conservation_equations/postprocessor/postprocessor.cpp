/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/acoustic_conservation_equations/postprocessor/postprocessor.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & postprocessor_data,
                                          MPI_Comm const &               comm)
  : mpi_comm(comm),
    pp_data(postprocessor_data),
    output_generator(comm),
    pointwise_output_generator(comm),
    error_calculator_p(comm),
    error_calculator_u(comm)

{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(AcousticsOperator const & pde_operator)
{
  output_generator.setup(pde_operator.get_dof_handler_p(),
                         pde_operator.get_dof_handler_u(),
                         *pde_operator.get_mapping(),
                         pp_data.output_data);

  pointwise_output_generator.setup(pde_operator.get_dof_handler_p(),
                                   pde_operator.get_dof_handler_u(),
                                   *pde_operator.get_mapping(),
                                   pp_data.pointwise_output_data);

  error_calculator_p.setup(pde_operator.get_dof_handler_p(),
                           *pde_operator.get_mapping(),
                           pp_data.error_data_p);

  error_calculator_u.setup(pde_operator.get_dof_handler_u(),
                           *pde_operator.get_mapping(),
                           pp_data.error_data_u);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(BlockVectorType const & solution,
                                              double const            time,
                                              types::time_step const  time_step_number)
{
  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    output_generator.evaluate(solution.block(block_index_pressure),
                              solution.block(block_index_velocity),
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  write pointwise output
   */
  if(pointwise_output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    pointwise_output_generator.evaluate(solution.block(block_index_pressure),
                                        solution.block(block_index_velocity),
                                        time,
                                        Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculate error
   */
  if(error_calculator_p.time_control.needs_evaluation(time, time_step_number))
    error_calculator_p.evaluate(solution.block(block_index_pressure),
                                time,
                                Utilities::is_unsteady_timestep(time_step_number));
  if(error_calculator_u.time_control.needs_evaluation(time, time_step_number))
    error_calculator_u.evaluate(solution.block(block_index_velocity),
                                time,
                                Utilities::is_unsteady_timestep(time_step_number));
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace Acoustics
} // namespace ExaDG
