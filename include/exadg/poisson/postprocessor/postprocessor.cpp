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

#include <exadg/poisson/postprocessor/postprocessor.h>
#include <exadg/poisson/spatial_discretization/operator.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                          MPI_Comm const &               mpi_comm_in)
  : mpi_comm(mpi_comm_in), pp_data(pp_data_in), output_generator(mpi_comm_in) //,
//    error_calculator(mpi_comm_in)
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(dealii::DoFHandler<dim, dim> const & dof_handler,
                                  dealii::Mapping<dim> const &         mapping)
{
  //  error_calculator.setup(dof_handler, mapping, pp_data.error_data);

  output_generator.setup(dof_handler, mapping, pp_data.output_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const & solution,
                                              double const       time,
                                              int const          time_step_number)
{
  //  error_calculator.evaluate(solution, time, time_step_number);

  output_generator.evaluate(solution, time, time_step_number);
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace Poisson
} // namespace ExaDG
