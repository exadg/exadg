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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_EVALUATE_FUNCTIONS_H_
#define EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_EVALUATE_FUNCTIONS_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>
#include <exadg/functions_and_boundary_conditions/function_with_normal.h>

// C/C++
#include <memory>

namespace ExaDG
{
template<int rank, int dim, typename Number>
struct FunctionEvaluator
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> const &                               function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points)
  {
    (void)function;
    (void)q_points;

    AssertThrow(false, dealii::ExcMessage("should not arrive here."));

    return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> &                                     function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    function.set_time(time);
    return value(function, q_points);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(ContainerInterfaceData<rank, dim, double> const & function,
          unsigned int const                                face,
          unsigned int const                                q,
          unsigned int const                                quad_index)
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
    value(FunctionWithNormal<dim> const &                                 function_with_normal,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const &     q_points,
          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normals)
  {
    (void)function_with_normal;
    (void)q_points;
    (void)normals;

    AssertThrow(false, dealii::ExcMessage("not implemented."));

    return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
    value(FunctionWithNormal<dim> &                                       function_with_normal,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const &     q_points,
          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normals,
          double const &                                                  time)
  {
    function_with_normal.set_time(time);
    return value(function_with_normal, q_points, normals);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::SymmetricTensor<rank, dim, dealii::VectorizedArray<Number>>
    value_symmetric(dealii::Function<dim> const &                               function,
                    dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points)
  {
    (void)function;
    (void)q_points;

    AssertThrow(false, dealii::ExcMessage("should not arrive here."));

    return dealii::SymmetricTensor<rank, dim, dealii::VectorizedArray<Number>>();
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::SymmetricTensor<rank, dim, dealii::VectorizedArray<Number>>
    value_symmetric(dealii::Function<dim> &                                     function,
                    dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
                    double const &                                              time)
  {
    function.set_time(time);
    return value_symmetric(function, q_points);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::SymmetricTensor<rank, dim, dealii::VectorizedArray<Number>>
    value_symmetric(ContainerInterfaceData<rank, dim, double> const & function,
                    unsigned int const                                face,
                    unsigned int const                                q,
                    unsigned int const                                quad_index)
  {
    (void)function;
    (void)face;
    (void)q;
    (void)quad_index;

    AssertThrow(false, dealii::ExcMessage("should not arrive here."));

    return dealii::SymmetricTensor<rank, dim, dealii::VectorizedArray<Number>>();
  }
};

template<int dim, typename Number>
struct FunctionEvaluator<0, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<0, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> const &                               function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points)
  {
    dealii::VectorizedArray<Number> value = dealii::make_vectorized_array<Number>(0.0);

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
    {
      dealii::Point<dim> q_point;
      for(unsigned int d = 0; d < dim; ++d)
        q_point[d] = q_points[d][v];

      value[v] = function.value(q_point);
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<0, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> &                                     function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    function.set_time(time);
    return value(function, q_points);
  }


  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<0, dim, dealii::VectorizedArray<Number>>
    value(ContainerInterfaceData<0, dim, double> const & function,
          unsigned int const                             face,
          unsigned int const                             q,
          unsigned int const                             quad_index)
  {
    dealii::VectorizedArray<Number> value = dealii::make_vectorized_array<Number>(0.0);

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      value[v] = function.get_data(quad_index, face, q, v);

    return value;
  }
};

template<int dim, typename Number>
struct FunctionEvaluator<1, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> const &                               function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        dealii::Point<dim> q_point;
        for(unsigned int i = 0; i < dim; ++i)
          q_point[i] = q_points[i][v];

        value[d][v] = function.value(q_point, d);
      }
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> &                                     function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    function.set_time(time);
    return value(function, q_points);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(ContainerInterfaceData<1, dim, double> const & function,
          unsigned int const                             face,
          unsigned int const                             q,
          unsigned int const                             quad_index)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> value;

    std::array<dealii::Tensor<1, dim, Number>, dealii::VectorizedArray<Number>::size()>
      tensor_array;

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      tensor_array[v] = function.get_data(quad_index, face, q, v);

    for(unsigned int d = 0; d < dim; ++d)
    {
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
        value[d][v] = tensor_array[v][d];
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(FunctionWithNormal<dim> &                                       function_with_normal,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const &     q_points,
          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normals)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> value;

    for(unsigned int d = 0; d < dim; ++d)
    {
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        dealii::Point<dim>     q_point;
        dealii::Tensor<1, dim> normal;
        for(unsigned int i = 0; i < dim; ++i)
        {
          q_point[i] = q_points[i][v];
          normal[i]  = normals[i][v];
        }
        function_with_normal.set_normal_vector(normal);
        value[d][v] = function_with_normal.value(q_point, d);
      }
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
    value(FunctionWithNormal<dim> &                                       function_with_normal,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const &     q_points,
          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & normals,
          double const &                                                  time)
  {
    function_with_normal.set_time(time);
    return value(function_with_normal, q_points, normals);
  }
};

template<int dim, typename Number>
struct FunctionEvaluator<2, dim, Number>
{
  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> const &                               function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points)
  {
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> value;

    for(unsigned int d1 = 0; d1 < dim; ++d1)
    {
      for(unsigned int d2 = 0; d2 < dim; ++d2)
      {
        for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
        {
          dealii::Point<dim> q_point;

          for(unsigned int i = 0; i < dim; ++i)
            q_point[i] = q_points[i][v];

          auto const unrolled_index =
            dealii::Tensor<2, dim>::component_to_unrolled_index(dealii::TableIndices<2>(d1, d2));

          value[d1][d2][v] = function.value(q_point, unrolled_index);
        }
      }
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
    value(dealii::Function<dim> &                                     function,
          dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
          double const &                                              time)
  {
    function.set_time(time);
    return value(function, q_points);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
    value(ContainerInterfaceData<2, dim, double> const & function,
          unsigned int const                             face,
          unsigned int const                             q,
          unsigned int const                             quad_index)
  {
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> value;

    std::array<dealii::Tensor<2, dim, Number>, dealii::VectorizedArray<Number>::size()>
      tensor_array;

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      tensor_array[v] = function.get_data(quad_index, face, q, v);

    for(unsigned int d1 = 0; d1 < dim; ++d1)
    {
      for(unsigned int d2 = 0; d2 < dim; ++d2)
      {
        for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
          value[d1][d2][v] = tensor_array[v][d1][d2];
      }
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
    value_symmetric(dealii::Function<dim> const &                               function,
                    dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points)
  {
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> value;

    for(unsigned int d1 = 0; d1 < dim; ++d1)
    {
      for(unsigned int d2 = d1; d2 < dim; ++d2)
      {
        for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
        {
          dealii::Point<dim> q_point;

          for(unsigned int i = 0; i < dim; ++i)
            q_point[i] = q_points[i][v];

          auto const unrolled_index = dealii::SymmetricTensor<2, dim>::component_to_unrolled_index(
            dealii::TableIndices<2>(d1, d2));

          value[d1][d2][v] = function.value(q_point, unrolled_index);
        }
      }
    }

    return value;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
    value_symmetric(dealii::Function<dim> &                                     function,
                    dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
                    double const &                                              time)
  {
    function.set_time(time);
    return value(q_points, time);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>
    value_symmetric(ContainerInterfaceData<2, dim, double> const & function,
                    unsigned int const                             face,
                    unsigned int const                             q,
                    unsigned int const                             quad_index)
  {
    dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> value;

    std::array<dealii::SymmetricTensor<2, dim, Number>, dealii::VectorizedArray<Number>::size()>
      tensor_array;

    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      tensor_array[v] = function.get_data(quad_index, face, q, v);

    for(unsigned int d1 = 0; d1 < dim; ++d1)
    {
      for(unsigned int d2 = d1; d2 < dim; ++d2)
      {
        for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
          value[d1][d2][v] = tensor_array[v][d1][d2];
      }
    }

    return value;
  }
};

} // namespace ExaDG

#endif /* EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_EVALUATE_FUNCTIONS_H_ */
