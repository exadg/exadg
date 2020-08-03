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

#include "function_cached.h"
#include "function_with_normal.h"

using namespace dealii;

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
    value(std::shared_ptr<FunctionCached<rank, dim>> function,
          unsigned int const                         face,
          unsigned int const                         q,
          unsigned int const                         quad_index)
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

    Number array[VectorizedArray<Number>::size()];
    for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
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
      value(std::shared_ptr<FunctionCached<0, dim>> function,
            unsigned int const                      face,
            unsigned int const                      q,
            unsigned int const                      quad_index)
  {
    VectorizedArray<Number> value = make_vectorized_array<Number>(0.0);

    Number array[VectorizedArray<Number>::size()];
    for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      array[v] = function->tensor_value(face, q, v, quad_index);
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
      Number array[VectorizedArray<Number>::size()];
      for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
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
      value(std::shared_ptr<FunctionCached<1, dim>> function,
            unsigned int const                      face,
            unsigned int const                      q,
            unsigned int const                      quad_index)
  {
    Tensor<1, dim, VectorizedArray<Number>> value;

    Tensor<1, dim, Number> tensor_array[VectorizedArray<Number>::size()];
    for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      tensor_array[v] = function->tensor_value(face, q, v, quad_index);
    }

    for(unsigned int d = 0; d < dim; ++d)
    {
      VectorizedArray<Number> array;
      for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
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
      Number array[VectorizedArray<Number>::size()];
      for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
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
