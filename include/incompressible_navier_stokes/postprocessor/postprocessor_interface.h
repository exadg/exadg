/*
 * postprocessor_interface.h
 *
 *  Created on: Feb 17, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_

#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace IncNS
{
namespace Interface
{
/*
 *  Interface for postprocessor of the incompressible Navier-Stokes equation.
 */
template<typename Number>
class PostProcessor
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual ~PostProcessor()
  {
  }

  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};

} // namespace Interface

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_ */
