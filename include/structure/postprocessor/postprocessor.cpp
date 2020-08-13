/*
 * postprocessor.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "postprocessor.h"

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                          MPI_Comm const &               mpi_comm_in)
  : pp_data(pp_data_in),
    mpi_comm(mpi_comm_in),
    output_generator(OutputGenerator<dim, Number>(mpi_comm_in)),
    error_calculator(ErrorCalculator<dim, Number>(mpi_comm_in))
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(DoFHandler<dim> const & dof_handler, Mapping<dim> const & mapping)
{
  output_generator.setup(dof_handler, mapping, pp_data.output_data);

  error_calculator.setup(dof_handler, mapping, pp_data.error_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const & solution,
                                              double const       time,
                                              int const          time_step_number)
{
  /*
   *  write output
   */
  output_generator.evaluate(solution, time, time_step_number);

  /*
   *  calculate error
   */
  error_calculator.evaluate(solution, time, time_step_number);
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace Structure
} // namespace ExaDG
