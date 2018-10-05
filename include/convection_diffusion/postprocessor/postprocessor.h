/*
 * PostProcessorConvDiff.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <sstream>

#include "convection_diffusion/postprocessor/output_generator.h"
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "postprocessor/calculate_l2_error.h"
#include "postprocessor/error_calculation.h"
#include "postprocessor/error_calculation_data.h"
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

template<int dim, int fe_degree>
class PostProcessor
{
public:
  typedef LinearAlgebra::distributed::Vector<double> VectorType;

  PostProcessor()
  {
  }

  void
  setup(PostProcessorData const & postprocessor_data,
        DoFHandler<dim> const &   dof_handler_in,
        Mapping<dim> const &      mapping_in,
        MatrixFree<dim, double> const & /*matrix_free_data_in*/,
        std::shared_ptr<ConvDiff::AnalyticalSolution<dim>> analytical_solution_in)
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
                    int const          time_step_number = -1);

private:
  ConvDiff::OutputGenerator<dim> output_generator;
  ErrorCalculator<dim, double>   error_calculator;
};

template<int dim, int fe_degree>
void
PostProcessor<dim, fe_degree>::do_postprocessing(VectorType const & solution,
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

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_ */
