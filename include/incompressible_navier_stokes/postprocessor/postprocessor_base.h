/*
 * postprocessor_base.h
 *
 *  Created on: Oct 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

#include "postprocessor_interface.h"

namespace IncNS
{
template<int dim, typename Number>
class DGNavierStokesBase;

/*
 *  Base class for postprocessor of the incompressible Navier-Stokes equation.
 */
template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
protected:
  typedef typename PostProcessorInterface<Number>::VectorType VectorType;

  typedef DGNavierStokesBase<dim, Number> Operator;

public:
  virtual ~PostProcessorBase()
  {
  }

  /*
   * Setup function.
   */
  virtual void
  setup(Operator const & pde_operator) = 0;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
