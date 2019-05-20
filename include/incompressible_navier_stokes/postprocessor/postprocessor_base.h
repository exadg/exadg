/*
 * postprocessor_base.h
 *
 *  Created on: Oct 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace IncNS
{
template<int dim, typename Number>
class DGNavierStokesBase;

/*
 *  Interface class for postprocessor of the
 *  incompressible Navier-Stokes equation.
 *
 */
template<int dim, typename Number>
class PostProcessorBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DGNavierStokesBase<dim, Number> NavierStokesOperator;

  PostProcessorBase()
  {
  }

  virtual ~PostProcessorBase()
  {
  }

  /*
   * Setup function.
   */
  virtual void
  setup(NavierStokesOperator const &    navier_stokes_operator,
        DoFHandler<dim> const &         dof_handler_velocity,
        DoFHandler<dim> const &         dof_handler_pressure,
        Mapping<dim> const &            mapping,
        MatrixFree<dim, Number> const & matrix_free_data) = 0;


  /*
   * This function has to be called to apply the postprocessing tools.
   */
  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
