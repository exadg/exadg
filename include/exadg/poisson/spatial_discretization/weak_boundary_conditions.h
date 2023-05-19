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

#ifndef INCLUDE_EXADG_POISSON_SPATIAL_DISCRETIZATION_WEAK_BOUNDARY_CONDITIONS_H_
#define INCLUDE_EXADG_POISSON_SPATIAL_DISCRETIZATION_WEAK_BOUNDARY_CONDITIONS_H_

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/operator_type.h>

namespace ExaDG
{
namespace Poisson
{
/*
 *  The following two functions calculate the interior_value/exterior_value
 *  depending on the operator type, the type of the boundary face
 *  and the given boundary conditions.
 *
 *                            +----------------------+--------------------+
 *                            | Dirichlet boundaries | Neumann boundaries |
 *  +-------------------------+----------------------+--------------------+
 *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
 *  +-------------------------+----------------------+--------------------+
 *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
 *  +-------------------------+----------------------+--------------------+
 *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
 *  +-------------------------+----------------------+--------------------+
 */
template<int dim, typename Number, int n_components, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_interior_value(unsigned int const                                q,
                           FaceIntegrator<dim, n_components, Number> const & integrator,
                           OperatorType const &                              operator_type)
{
  if(operator_type == OperatorType::full or operator_type == OperatorType::homogeneous)
  {
    return integrator.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
  }

  return dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>();
}

template<int dim, typename Number, int n_components, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_exterior_value(
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> const & value_m,
    unsigned int const                                                 q,
    FaceIntegrator<dim, n_components, Number> const &                  integrator,
    OperatorType const &                                               operator_type,
    BoundaryType const &                                               boundary_type,
    dealii::types::boundary_id const                                   boundary_id,
    std::shared_ptr<BoundaryDescriptor<rank, dim> const>               boundary_descriptor,
    double const &                                                     time)
{
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> value_p;

  if(boundary_type == BoundaryType::Dirichlet or boundary_type == BoundaryType::DirichletCached)
  {
    if(operator_type == OperatorType::full or operator_type == OperatorType::inhomogeneous)
    {
      dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> g;

      if(boundary_type == BoundaryType::Dirichlet)
      {
        auto bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
        auto q_points = integrator.quadrature_point(q);

        g = FunctionEvaluator<rank, dim, Number>::value(*bc, q_points, time);
      }
      else if(boundary_type == BoundaryType::DirichletCached)
      {
        auto bc = boundary_descriptor->get_dirichlet_cached_data();
        g       = FunctionEvaluator<rank, dim, Number>::value(*bc,
                                                        integrator.get_current_cell_index(),
                                                        q,
                                                        integrator.get_quadrature_index());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }

      value_p = -value_m + dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>(2.0 * g);
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      value_p = -value_m;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else if(boundary_type == BoundaryType::Neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

// clang-format off
  /*
   *  The following two functions calculate the interior/exterior gradient
   *  in normal direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n + 2h                    |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n                         |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁺)*n = grad(phi⁻)*n, grad(phi⁻)*n = 0 | grad(phi⁺)*n = -grad(phi⁻)*n + 2h, grad(phi⁻)*n  = 0 |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = 0                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                           | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   */
// clang-format on
template<int dim, typename Number, int n_components, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_interior_normal_gradient(unsigned int const                                q,
                                     FaceIntegrator<dim, n_components, Number> const & integrator,
                                     OperatorType const & operator_type)
{
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> normal_gradient_m;

  if(operator_type == OperatorType::full or operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = integrator.get_normal_derivative(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    // do nothing (normal_gradient_m already initialized with 0.0)
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
  }

  return normal_gradient_m;
}

template<int dim, typename Number, int n_components, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_exterior_normal_gradient(
    dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> const & normal_gradient_m,
    unsigned int const                                                 q,
    FaceIntegrator<dim, n_components, Number> const &                  integrator,
    OperatorType const &                                               operator_type,
    BoundaryType const &                                               boundary_type,
    dealii::types::boundary_id const                                   boundary_id,
    std::shared_ptr<BoundaryDescriptor<rank, dim> const>               boundary_descriptor,
    double const &                                                     time)
{
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> normal_gradient_p;

  if(boundary_type == BoundaryType::Dirichlet or boundary_type == BoundaryType::DirichletCached)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::Neumann)
  {
    if(operator_type == OperatorType::full or operator_type == OperatorType::inhomogeneous)
    {
      auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
      auto q_points = integrator.quadrature_point(q);

      auto h = FunctionEvaluator<rank, dim, Number>::value(*bc, q_points, time);

      normal_gradient_p =
        -normal_gradient_m + dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>(2.0 * h);
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      normal_gradient_p = -normal_gradient_m;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
}

/*
 * This function calculates the Neumann boundary value and is required in case of continuous
 * Galerkin discretizations.
 */
template<int dim, typename Number, int n_components, int rank>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>>
  calculate_neumann_value(unsigned int const                                   q,
                          FaceIntegrator<dim, n_components, Number> const &    integrator,
                          BoundaryType const &                                 boundary_type,
                          dealii::types::boundary_id const                     boundary_id,
                          std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor,
                          double const &                                       time)
{
  dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> normal_gradient;

  if(boundary_type == BoundaryType::Neumann)
  {
    auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
    auto q_points = integrator.quadrature_point(q);

    normal_gradient = FunctionEvaluator<rank, dim, Number>::value(*bc, q_points, time);
  }
  else
  {
    // do nothing

    Assert(boundary_type == BoundaryType::Dirichlet or
             boundary_type == BoundaryType::DirichletCached,
           dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient;
}

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_EXADG_POISSON_SPATIAL_DISCRETIZATION_WEAK_BOUNDARY_CONDITIONS_H_ */
