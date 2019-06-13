/*
 * weak_boundary_conditions.h
 *
 *  Created on: Jun 4, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/operator_type.h"
#include "../../user_interface/boundary_descriptor.h"

namespace ConvDiff
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
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_interior_value(unsigned int const                     q,
                           FaceIntegrator<dim, 1, Number> const & integrator,
                           OperatorType const &                   operator_type)
{
  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    return integrator.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    return make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return make_vectorized_array<Number>(0.0);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_exterior_value(VectorizedArray<Number> const &                value_m,
                           unsigned int const                             q,
                           FaceIntegrator<dim, 1, Number> const &         integrator,
                           OperatorType const &                           operator_type,
                           BoundaryType const &                           boundary_type,
                           types::boundary_id const                       boundary_id,
                           std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor,
                           double const &                                 time)
{
  VectorizedArray<Number> value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        boundary_descriptor->dirichlet_bc.find(boundary_id);
      Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

      VectorizedArray<Number> g = evaluate_scalar_function(it->second, q_points, time);

      value_p = -value_m + 2.0 * g;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      value_p = -value_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

// clang-format off
  /*
   *  The following two functions calculate the interior/exterior velocity gradient
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
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_interior_normal_gradient(unsigned int const                     q,
                                     FaceIntegrator<dim, 1, Number> const & integrator,
                                     OperatorType const &                   operator_type)
{
  VectorizedArray<Number> normal_gradient_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = integrator.get_normal_derivative(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    normal_gradient_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return normal_gradient_m;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_exterior_normal_gradient(
    VectorizedArray<Number> const &                normal_gradient_m,
    unsigned int const                             q,
    FaceIntegrator<dim, 1, Number> const &         integrator,
    OperatorType const &                           operator_type,
    BoundaryType const &                           boundary_type,
    types::boundary_id const                       boundary_id,
    std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor,
    double const &                                 time)
{
  VectorizedArray<Number> normal_gradient_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        boundary_descriptor->neumann_bc.find(boundary_id);
      Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

      VectorizedArray<Number> h = evaluate_scalar_function(it->second, q_points, time);

      normal_gradient_p = -normal_gradient_m + 2.0 * h;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      normal_gradient_p = -normal_gradient_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
}

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_ \
        */
