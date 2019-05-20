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
// forward declaration
template<int dim, typename Number>
class DGOperator;

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
  setup(DGOperator<dim, Number> const & navier_stokes_operator_in,
        DoFHandler<dim> const &         dof_handler_in,
        DoFHandler<dim> const &         dof_handler_vector_in,
        DoFHandler<dim> const &         dof_handler_scalar_in,
        Mapping<dim> const &            mapping_in,
        MatrixFree<dim, Number> const & matrix_free_data_in) = 0;

  virtual void
  do_postprocessing(VectorType const & solution, double const time, int const time_step_number) = 0;
};

} // namespace CompNS



#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
