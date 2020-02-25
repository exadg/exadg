/*
 * postprocessor_base.h
 *
 *  Created on: May 16, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace CompNS
{
template<typename Number>
class PostProcessorInterface
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual ~PostProcessorInterface()
  {
  }

  virtual void
  do_postprocessing(VectorType const & solution, double const time, int const time_step_number) = 0;
};

// forward declaration
template<int dim, typename Number>
class DGOperator;

template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
private:
  typedef typename PostProcessorInterface<Number>::VectorType VectorType;

public:
  virtual ~PostProcessorBase()
  {
  }

  virtual void
  setup(DGOperator<dim, Number> const & pde_operator) = 0;
};

} // namespace CompNS



#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
