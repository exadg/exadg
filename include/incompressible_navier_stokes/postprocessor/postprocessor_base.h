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
 *  Base class for postprocessor of the incompressible Navier-Stokes equation.
 */
template<int dim, typename Number>
class PostProcessorBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DGNavierStokesBase<dim, Number> Operator;

  virtual ~PostProcessorBase()
  {
  }

  /*
   * Setup function.
   */
  virtual void
  setup(Operator const & pde_operator) = 0;


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
