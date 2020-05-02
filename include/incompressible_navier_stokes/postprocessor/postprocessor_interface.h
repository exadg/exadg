/*
 * postprocessor_interface.h
 *
 *  Created on: 02.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

namespace IncNS
{
template<typename Number>
class PostProcessorInterface
{
protected:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorInterface()
  {
  }

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


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_ */
