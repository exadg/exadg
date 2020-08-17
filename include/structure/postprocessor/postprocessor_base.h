/*
 * postprocessor_base.h
 *
 *  Created on: 02.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<typename Number>
class PostProcessorBase
{
protected:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorBase()
  {
  }

  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) = 0;
};

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
