/*
 * evaluate_functions.h
 *
 *  Created on: Nov 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EVALUATEFUNCTIONS_H_
#define INCLUDE_EVALUATEFUNCTIONS_H_

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include "../../include/functionalities/function_with_normal.h"

using namespace dealii;

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<value_type>
  evaluate_scalar_function(std::shared_ptr<Function<dim>>                  function,
                           Point<dim, VectorizedArray<value_type>> const & q_points,
                           double const &                                  eval_time)
{
  VectorizedArray<value_type> value = make_vectorized_array<value_type>(0.0);

  value_type array[VectorizedArray<value_type>::n_array_elements];
  for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
  {
    Point<dim> q_point;
    for(unsigned int d = 0; d < dim; ++d)
      q_point[d] = q_points[d][n];

    function->set_time(eval_time);
    array[n] = function->value(q_point);
  }
  value.load(&array[0]);

  return value;
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<value_type>>
  evaluate_vectorial_function(std::shared_ptr<Function<dim>>                  function,
                              Point<dim, VectorizedArray<value_type>> const & q_points,
                              double const &                                  eval_time)
{
  Tensor<1, dim, VectorizedArray<value_type>> value;

  for(unsigned int d = 0; d < dim; ++d)
  {
    value_type array[VectorizedArray<value_type>::n_array_elements];
    for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
    {
      Point<dim> q_point;
      for(unsigned int d = 0; d < dim; ++d)
        q_point[d] = q_points[d][n];

      function->set_time(eval_time);
      array[n] = function->value(q_point, d);
    }
    value[d].load(&array[0]);
  }

  return value;
}

template<int dim, typename value_type>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<value_type>>
  evaluate_vectorial_function_with_normal(
    std::shared_ptr<Function<dim>>                      function_in,
    Point<dim, VectorizedArray<value_type>> const &     q_points,
    Tensor<1, dim, VectorizedArray<value_type>> const & normals_m,
    double const &                                      eval_time)
{
  auto function = std::dynamic_pointer_cast<FunctionWithNormal<dim>>(function_in);

  Tensor<1, dim, VectorizedArray<value_type>> value;

  for(unsigned int d = 0; d < dim; ++d)
  {
    value_type array[VectorizedArray<value_type>::n_array_elements];
    for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
    {
      Point<dim>     q_point;
      Tensor<1, dim> normal_m;
      for(unsigned int d = 0; d < dim; ++d)
      {
        q_point[d]  = q_points[d][n];
        normal_m[d] = normals_m[d][n];
      }
      function->set_time(eval_time);
      function->set_normal_vector(normal_m);
      array[n] = function->value(q_point, d);
    }
    value[d].load(&array[0]);
  }

  return value;
}

#endif /* INCLUDE_EVALUATEFUNCTIONS_H_ */
