/*
 * postprocessor_base.h
 *
 *  Created on: Oct 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_interface.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
class SpatialOperatorBase;

/*
 *  Base class for postprocessor of the incompressible Navier-Stokes equation.
 */
template<int dim, typename Number>
class PostProcessorBase : public PostProcessorInterface<Number>
{
protected:
  typedef typename PostProcessorInterface<Number>::VectorType VectorType;

  typedef SpatialOperatorBase<dim, Number> Operator;

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
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
