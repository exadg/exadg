#ifndef INCLUDE_FUNCTIONALITIES_SET_ZERO_MEAN_VALUE_H_
#define INCLUDE_FUNCTIONALITIES_SET_ZERO_MEAN_VALUE_H_

#include <deal.II/lac/parallel_vector.h>

template<typename Number>
void
set_zero_mean_value(parallel::distributed::Vector<Number> & vec)
{
  const Number mean_val = vec.mean_value();
  vec.add(-mean_val);
}

#endif