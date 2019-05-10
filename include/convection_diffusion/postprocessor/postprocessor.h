/*
 * postprocessor.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <sstream>

#include "convection_diffusion/postprocessor/output_generator.h"
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "postprocessor/error_calculation.h"
#include "postprocessor/output_data.h"

namespace ConvDiff
{
struct PostProcessorData
{
  PostProcessorData()
  {
  }

  OutputData           output_data;
  ErrorCalculationData error_data;
};

template<int dim, typename Number>
class PostProcessor
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  void
  setup(PostProcessorData const & postprocessor_data,
        DoFHandler<dim> const &   dof_handler_in,
        Mapping<dim> const &      mapping_in,
        MatrixFree<dim, Number> const & /*matrix_free_data_in*/,
        std::shared_ptr<ConvDiff::AnalyticalSolution<dim>> const analytical_solution_in)
  {
    error_calculator.setup(dof_handler_in,
                           mapping_in,
                           analytical_solution_in->solution,
                           postprocessor_data.error_data);

    output_generator.setup(dof_handler_in, mapping_in, postprocessor_data.output_data);
  }

  void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1)
  {
    error_calculator.evaluate(solution, time, time_step_number);

    output_generator.evaluate(solution, time, time_step_number);
  }

private:
  ConvDiff::OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number>           error_calculator;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_ */
