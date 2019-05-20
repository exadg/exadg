/*
 * postprocessor_base.h
 *
 *  Created on: May 14, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "convection_diffusion/user_interface/analytical_solution.h"

using namespace dealii;

namespace ConvDiff
{
template<int dim, typename Number>
class PostProcessorBase
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorBase()
  {
  }

  virtual void
  setup(DoFHandler<dim> const &              dof_handler_in,
        Mapping<dim> const &                 mapping_in,
        MatrixFree<dim, Number> const &      matrix_free_data_in,
        std::shared_ptr<Function<dim>> const analytical_solution_in) = 0;

  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
