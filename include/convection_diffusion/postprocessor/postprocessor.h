/*
 * postprocessor.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

#include "convection_diffusion/user_interface/analytical_solution.h"
#include "output_generator.h"
#include "postprocessor/error_calculation.h"
#include "postprocessor/output_data.h"
#include "postprocessor_base.h"

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
class PostProcessor : public PostProcessorBase<dim, Number>
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  PostProcessor(PostProcessorData const & pp_data_in);

  void
  setup(DoFHandler<dim> const &              dof_handler_in,
        Mapping<dim> const &                 mapping_in,
        MatrixFree<dim, Number> const &      matrix_free_data_in,
        std::shared_ptr<Function<dim>> const analytical_solution_in);

  void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1);

private:
  PostProcessorData pp_data;

  ConvDiff::OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number>           error_calculator;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_ */
