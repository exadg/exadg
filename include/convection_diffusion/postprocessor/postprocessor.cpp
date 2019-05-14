/*
 * postprocessor.cpp
 *
 *  Created on: May 13, 2019
 *      Author: fehn
 */

#include "postprocessor.h"

namespace ConvDiff
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData const & pp_data_in)
  : pp_data(pp_data_in)
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(DoFHandler<dim> const & dof_handler_in,
                                  Mapping<dim> const &    mapping_in,
                                  MatrixFree<dim, Number> const & /*matrix_free_data_in*/,
                                  std::shared_ptr<Function<dim>> const analytical_solution_in)
{
  error_calculator.setup(dof_handler_in, mapping_in, analytical_solution_in, pp_data.error_data);

  output_generator.setup(dof_handler_in, mapping_in, pp_data.output_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const & solution,
                                              double const       time,
                                              int const          time_step_number)
{
  error_calculator.evaluate(solution, time, time_step_number);

  output_generator.evaluate(solution, time, time_step_number);
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace ConvDiff
