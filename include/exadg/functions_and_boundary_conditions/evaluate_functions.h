/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EVALUATEFUNCTIONS_H_
#define INCLUDE_EVALUATEFUNCTIONS_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <exadg/functions_and_boundary_conditions/function_cached.h>
#include <exadg/functions_and_boundary_conditions/function_with_normal.h>

namespace ExaDG
{
template<int rank, int dim, typename Number>
struct FunctionEvaluator
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<dealii::Function<dim>>                      function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    (void)function;
    (void)q_points;
    (void)time;

    AssertThrow(false, dealii::ExcMessage("should not arrive here."));

    return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<FunctionCached<rank, dim>> function,
          unsigned int const                         face,
          unsigned int const                         q,
          unsigned int const                         quad_index)
  {
    (void)function;
    (void)face;
    (void)q;
    (void)quad_index;

    AssertThrow(false, dealii::ExcMessage("should not arrive here."));

    return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<dealii::Function<dim>>                          function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const &     q_points,
          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normals,
          double const &                                                  time)
  {
    (void)function;
    (void)q_points;
    (void)normals;
    (void)time;

    AssertThrow(false, dealii::ExcMessage("not implemented."));

    return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
  }
};

template<int dim, typename Number>
struct FunctionEvaluator<0, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<0, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<dealii::Function<dim>>                      function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    dealii::VectorizedArray<Number> value = dealii::make_vectorized_array<Number>(0.0);

    Number array[dealii::VectorizedArray<Number>::size()];
    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
    {
      dealii::Point<dim> q_point;
      for(unsigned int d = 0; d < dim; ++d)
        q_point[d] = q_points[d][v];

      function->set_time(time);
      array[v] = function->value(q_point);
    }
    value.load(&array[0]);

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<0, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<FunctionCached<0, dim>> function,
          unsigned int const                      face,
          unsigned int const                      q,
          unsigned int const                      quad_index)
  {
    dealii::VectorizedArray<Number> value = dealii::make_vectorized_array<Number>(0.0);

    Number array[dealii::VectorizedArray<Number>::size()];
    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
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
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<dealii::Function<dim>>                      function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      Number array[dealii::VectorizedArray<Number>::size()];
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        dealii::Point<dim> q_point;
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
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<FunctionCached<1, dim>> function,
          unsigned int const                      face,
          unsigned int const                      q,
          unsigned int const                      quad_index)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> value;

    dealii::Tensor<1, dim, Number> tensor_array[dealii::VectorizedArray<Number>::size()];
    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
    {
      tensor_array[v] = function->tensor_value(face, q, v, quad_index);
    }

    for(unsigned int d = 0; d < dim; ++d)
    {
      dealii::VectorizedArray<Number> array;
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
        array[v] = tensor_array[v][d];

      value[d].load(&array[0]);
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(std::shared_ptr<dealii::Function<dim>>                          function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const &     q_points,
          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normals,
          double const &                                                  time)
  {
    auto function_with_normal = std::dynamic_pointer_cast<FunctionWithNormal<dim>>(function);

    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      Number array[dealii::VectorizedArray<Number>::size()];
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        dealii::Point<dim>     q_point;
        dealii::Tensor<1, dim> normal;
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

} // namespace ExaDG

#endif /* INCLUDE_EVALUATEFUNCTIONS_H_ */
