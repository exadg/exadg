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

#include <exadg/convection_diffusion/postprocessor/postprocessor.h>
#include <exadg/convection_diffusion/spatial_discretization/operator.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                          MPI_Comm const &               mpi_comm_in)
  : mpi_comm(mpi_comm_in),
    pp_data(pp_data_in),
    output_generator(mpi_comm_in),
    error_calculator(mpi_comm_in)
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator<dim, Number> const & pde_operator)
{
  error_calculator.setup(pde_operator.get_dof_handler(),
                         *pde_operator.get_mapping(),
                         pp_data.error_data);

  output_generator.setup(pde_operator.get_dof_handler(),
                         *pde_operator.get_mapping(),
                         pp_data.output_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup_after_coarsening_and_refinement()
{
  // The `error_calculator` and `output_generator` do not require any additional setup after
  // coarsening and refinement.
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     solution,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  if(error_calculator.time_control.needs_evaluation(time, time_step_number))
    error_calculator.evaluate(solution, time, Utilities::is_unsteady_timestep(time_step_number));

  if(output_generator.time_control.needs_evaluation(time, time_step_number))
    output_generator.evaluate(solution, time, Utilities::is_unsteady_timestep(time_step_number));
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
