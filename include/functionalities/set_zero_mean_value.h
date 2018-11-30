#ifndef INCLUDE_FUNCTIONALITIES_SET_ZERO_MEAN_VALUE_H_
#define INCLUDE_FUNCTIONALITIES_SET_ZERO_MEAN_VALUE_H_

#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

template<typename Number>
void
set_zero_mean_value(LinearAlgebra::distributed::Vector<Number> & vector)
{
  Number const mean_value = vector.mean_value();
  vector.add(-mean_value);
}

#endif
