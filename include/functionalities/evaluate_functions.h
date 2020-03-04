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

// template<int dim, typename value_type>
// inline DEAL_II_ALWAYS_INLINE //
//  VectorizedArray<value_type>
//  evaluate_scalar_function(std::shared_ptr<Function<dim>>                  function,
//                           Point<dim, VectorizedArray<value_type>> const & q_points,
//                           double const &                                  time)
//{
//  VectorizedArray<value_type> value = make_vectorized_array<value_type>(0.0);
//
//  value_type array[VectorizedArray<value_type>::n_array_elements];
//  for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
//  {
//    Point<dim> q_point;
//    for(unsigned int d = 0; d < dim; ++d)
//      q_point[d] = q_points[d][n];
//
//    function->set_time(time);
//    array[n] = function->value(q_point);
//  }
//  value.load(&array[0]);
//
//  return value;
//}
//
// template<int dim, typename value_type>
// inline DEAL_II_ALWAYS_INLINE //
//  Tensor<1, dim, VectorizedArray<value_type>>
//  evaluate_vectorial_function(std::shared_ptr<Function<dim>>                  function,
//                              Point<dim, VectorizedArray<value_type>> const & q_points,
//                              double const &                                  time)
//{
//  Tensor<1, dim, VectorizedArray<value_type>> value;
//
//  for(unsigned int d = 0; d < dim; ++d)
//  {
//    value_type array[VectorizedArray<value_type>::n_array_elements];
//    for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
//    {
//      Point<dim> q_point;
//      for(unsigned int d = 0; d < dim; ++d)
//        q_point[d] = q_points[d][n];
//
//      function->set_time(time);
//      array[n] = function->value(q_point, d);
//    }
//    value[d].load(&array[0]);
//  }
//
//  return value;
//}
//
// template<int dim, typename value_type, int rank>
// inline DEAL_II_ALWAYS_INLINE //
//  Tensor<rank, dim, VectorizedArray<value_type>>
//  evaluate_function(std::shared_ptr<Function<dim>>                  function,
//                    Point<dim, VectorizedArray<value_type>> const & q_points,
//                    double const &                                  time)
//{
//  (void)function;
//  (void)q_points;
//  (void)time;
//
//  return Tensor<rank, dim, VectorizedArray<value_type>>();
//
//  if constexpr(rank == 0)
//    return evaluate_scalar_function<dim, value_type>(function, q_points, time);
//  else
//    return evaluate_vectorial_function<dim, value_type>(function, q_points, time);
//}

template<int dim, typename value_type, int rank>
struct FunctionEvaluator
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<rank, dim, VectorizedArray<value_type>>
    value(std::shared_ptr<Function<dim>>                  function,
          Point<dim, VectorizedArray<value_type>> const & q_points,
          double const &                                  time)
  {
    (void)function;
    (void)q_points;
    (void)time;

    AssertThrow(false, ExcMessage("should not arrive here."));

    return Tensor<rank, dim, VectorizedArray<value_type>>();
  }


  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<rank, dim, VectorizedArray<value_type>>
    value(std::shared_ptr<Function<dim>>                      function,
          Point<dim, VectorizedArray<value_type>> const &     q_points,
          Tensor<1, dim, VectorizedArray<value_type>> const & normals,
          double const &                                      time)
  {
    (void)function;
    (void)q_points;
    (void)normals;
    (void)time;

    AssertThrow(false, ExcMessage("not implemented."));

    return Tensor<rank, dim, VectorizedArray<value_type>>();
  }
};

template<int dim, typename value_type>
struct FunctionEvaluator<dim, value_type, 0>
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<0, dim, VectorizedArray<value_type>>
    value(std::shared_ptr<Function<dim>>                  function,
          Point<dim, VectorizedArray<value_type>> const & q_points,
          double const &                                  time)
  {
    VectorizedArray<value_type> value = make_vectorized_array<value_type>(0.0);

    value_type array[VectorizedArray<value_type>::n_array_elements];
    for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
    {
      Point<dim> q_point;
      for(unsigned int d = 0; d < dim; ++d)
        q_point[d] = q_points[d][n];

      function->set_time(time);
      array[n] = function->value(q_point);
    }
    value.load(&array[0]);

    return value;
  }
};

template<int dim, typename value_type>
struct FunctionEvaluator<dim, value_type, 1>
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<value_type>>
    value(std::shared_ptr<Function<dim>>                  function,
          Point<dim, VectorizedArray<value_type>> const & q_points,
          double const &                                  time)
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

        function->set_time(time);
        array[n] = function->value(q_point, d);
      }
      value[d].load(&array[0]);
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<value_type>>
    value(std::shared_ptr<Function<dim>>                      function,
          Point<dim, VectorizedArray<value_type>> const &     q_points,
          Tensor<1, dim, VectorizedArray<value_type>> const & normals,
          double const &                                      time)
  {
    auto function_with_normal = std::dynamic_pointer_cast<FunctionWithNormal<dim>>(function);

    Tensor<1, dim, VectorizedArray<value_type>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      value_type array[VectorizedArray<value_type>::n_array_elements];
      for(unsigned int n = 0; n < VectorizedArray<value_type>::n_array_elements; ++n)
      {
        Point<dim>     q_point;
        Tensor<1, dim> normal;
        for(unsigned int d = 0; d < dim; ++d)
        {
          q_point[d] = q_points[d][n];
          normal[d]  = normals[d][n];
        }
        function_with_normal->set_time(time);
        function_with_normal->set_normal_vector(normal);
        array[n] = function_with_normal->value(q_point, d);
      }
      value[d].load(&array[0]);
    }

    return value;
  }
};

#endif /* INCLUDE_EVALUATEFUNCTIONS_H_ */
