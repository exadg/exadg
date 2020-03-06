/*
 * evaluate_functions.h
 *
 *  Created on: Nov 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EVALUATEFUNCTIONS_H_
#define INCLUDE_EVALUATEFUNCTIONS_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include "../../include/functionalities/function_interpolation.h"
#include "../../include/functionalities/function_with_normal.h"

using namespace dealii;

// template<int dim, typename Number>
// inline DEAL_II_ALWAYS_INLINE //
//  VectorizedArray<Number>
//  evaluate_scalar_function(std::shared_ptr<Function<dim>>                  function,
//                           Point<dim, VectorizedArray<Number>> const & q_points,
//                           double const &                                  time)
//{
//  VectorizedArray<Number> value = make_vectorized_array<Number>(0.0);
//
//  Number array[VectorizedArray<Number>::n_array_elements];
//  for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
//  {
//    Point<dim> q_point;
//    for(unsigned int d = 0; d < dim; ++d)
//      q_point[d] = q_points[d][v];
//
//    function->set_time(time);
//    array[v] = function->value(q_point);
//  }
//  value.load(&array[0]);
//
//  return value;
//}
//
// template<int dim, typename Number>
// inline DEAL_II_ALWAYS_INLINE //
//  Tensor<1, dim, VectorizedArray<Number>>
//  evaluate_vectorial_function(std::shared_ptr<Function<dim>>                  function,
//                              Point<dim, VectorizedArray<Number>> const & q_points,
//                              double const &                                  time)
//{
//  Tensor<1, dim, VectorizedArray<Number>> value;
//
//  for(unsigned int d = 0; d < dim; ++d)
//  {
//    Number array[VectorizedArray<Number>::n_array_elements];
//    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
//    {
//      Point<dim> q_point;
//      for(unsigned int d = 0; d < dim; ++d)
//        q_point[d] = q_points[d][v];
//
//      function->set_time(time);
//      array[v] = function->value(q_point, d);
//    }
//    value[d].load(&array[0]);
//  }
//
//  return value;
//}
//
// template<int dim, typename Number, int rank>
// inline DEAL_II_ALWAYS_INLINE //
//  Tensor<rank, dim, VectorizedArray<Number>>
//  evaluate_function(std::shared_ptr<Function<dim>>                  function,
//                    Point<dim, VectorizedArray<Number>> const & q_points,
//                    double const &                                  time)
//{
//  (void)function;
//  (void)q_points;
//  (void)time;
//
//  return Tensor<rank, dim, VectorizedArray<Number>>();
//
//  if constexpr(rank == 0)
//    return evaluate_scalar_function<dim, Number>(function, q_points, time);
//  else
//    return evaluate_vectorial_function<dim, Number>(function, q_points, time);
//}

template<int rank, int dim, typename Number>
struct FunctionEvaluator
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<rank, dim, VectorizedArray<Number>>
    value(std::shared_ptr<Function<dim>>              function,
          Point<dim, VectorizedArray<Number>> const & q_points,
          double const &                              time)
  {
    (void)function;
    (void)q_points;
    (void)time;

    AssertThrow(false, ExcMessage("should not arrive here."));

    return Tensor<rank, dim, VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<rank, dim, VectorizedArray<Number>>
    value(std::shared_ptr<FunctionInterpolation<rank, dim, Number>> function,
          unsigned int const                                        face,
          unsigned int const                                        q,
          unsigned int const                                        quad_index)
  {
    (void)function;
    (void)face;
    (void)q;
    (void)quad_index;

    AssertThrow(false, ExcMessage("should not arrive here."));

    return Tensor<rank, dim, VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<rank, dim, VectorizedArray<Number>>
    value(std::shared_ptr<Function<dim>>                  function,
          Point<dim, VectorizedArray<Number>> const &     q_points,
          Tensor<1, dim, VectorizedArray<Number>> const & normals,
          double const &                                  time)
  {
    (void)function;
    (void)q_points;
    (void)normals;
    (void)time;

    AssertThrow(false, ExcMessage("not implemented."));

    return Tensor<rank, dim, VectorizedArray<Number>>();
  }
};

template<int dim, typename Number>
struct FunctionEvaluator<0, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<0, dim, VectorizedArray<Number>>
    value(std::shared_ptr<Function<dim>>              function,
          Point<dim, VectorizedArray<Number>> const & q_points,
          double const &                              time)
  {
    VectorizedArray<Number> value = make_vectorized_array<Number>(0.0);

    Number array[VectorizedArray<Number>::n_array_elements];
    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      Point<dim> q_point;
      for(unsigned int d = 0; d < dim; ++d)
        q_point[d] = q_points[d][v];

      function->set_time(time);
      array[v] = function->value(q_point);
    }
    value.load(&array[0]);

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
      Tensor<0, dim, VectorizedArray<Number>>
      value(std::shared_ptr<FunctionInterpolation<0, dim, Number>> function,
            unsigned int const                                     face,
            unsigned int const                                     q,
            unsigned int const                                     quad_index)
  {
    VectorizedArray<Number> value = make_vectorized_array<Number>(0.0);

    Number array[VectorizedArray<Number>::n_array_elements];
    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      array[v] = function->value(face, q, v, quad_index);
    }
    value.load(&array[0]);

    return value;
  }
};

template<int dim, typename Number>
struct FunctionEvaluator<1, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<Number>>
    value(std::shared_ptr<Function<dim>>              function,
          Point<dim, VectorizedArray<Number>> const & q_points,
          double const &                              time)
  {
    Tensor<1, dim, VectorizedArray<Number>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      Number array[VectorizedArray<Number>::n_array_elements];
      for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
      {
        Point<dim> q_point;
        for(unsigned int d = 0; d < dim; ++d)
          q_point[d] = q_points[d][v];

        function->set_time(time);
        array[v] = function->value(q_point, d);
      }
      value[d].load(&array[0]);
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, VectorizedArray<Number>>
      value(std::shared_ptr<FunctionInterpolation<1, dim, Number>> function,
            unsigned int const                                     face,
            unsigned int const                                     q,
            unsigned int const                                     quad_index)
  {
    Tensor<1, dim, VectorizedArray<Number>> value;

    Tensor<1, dim, Number> tensor_array[VectorizedArray<Number>::n_array_elements];
    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      tensor_array[v] = function->value(face, q, v, quad_index);
    }

    for(unsigned int d = 0; d < dim; ++d)
    {
      VectorizedArray<Number> array;
      for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        array[v] = tensor_array[v][d];

      value[d].load(&array[0]);
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<Number>>
    value(std::shared_ptr<Function<dim>>                  function,
          Point<dim, VectorizedArray<Number>> const &     q_points,
          Tensor<1, dim, VectorizedArray<Number>> const & normals,
          double const &                                  time)
  {
    auto function_with_normal = std::dynamic_pointer_cast<FunctionWithNormal<dim>>(function);

    Tensor<1, dim, VectorizedArray<Number>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      Number array[VectorizedArray<Number>::n_array_elements];
      for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
      {
        Point<dim>     q_point;
        Tensor<1, dim> normal;
        for(unsigned int d = 0; d < dim; ++d)
        {
          q_point[d] = q_points[d][v];
          normal[d]  = normals[d][v];
        }
        function_with_normal->set_time(time);
        function_with_normal->set_normal_vector(normal);
        array[v] = function_with_normal->value(q_point, d);
      }
      value[d].load(&array[0]);
    }

    return value;
  }
};

#endif /* INCLUDE_EVALUATEFUNCTIONS_H_ */
