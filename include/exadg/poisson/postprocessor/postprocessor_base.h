/*
 * postprocessor_base.h
 *
 *  Created on: May 14, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_POISSON_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_EXADG_POISSON_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<typename Number>
class PostProcessorInterface
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual ~PostProcessorInterface()
  {
  }

  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};

template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
protected:
  typedef typename PostProcessorInterface<Number>::VectorType VectorType;

public:
  virtual ~PostProcessorBase()
  {
  }

  virtual void
  setup(DoFHandler<dim, dim> const & dof_handler, Mapping<dim> const & mapping) = 0;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_EXADG_POISSON_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
