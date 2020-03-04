/*
 * weak_boundary_conditions.h
 *
 *  Created on: Jun 13, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/operator_type.h"
#include "../../user_interface/boundary_descriptor.h"

namespace IncNS
{
// clang-format off
/*
 * Velocity:
 *
 *  The following two functions calculate the interior/exterior value for boundary faces depending on the
 *  operator type, the type of the boundary face and the given boundary conditions.
 *
 *                            +-------------------------+--------------------+------------------------------+
 *                            | Dirichlet boundaries    | Neumann boundaries | symmetry boundaries          |
 *  +-------------------------+-------------------------+--------------------+------------------------------+
 *  | full operator           | u⁺ = -u⁻ + 2g           | u⁺ = u⁻            | u⁺ = u⁻ - 2 (u⁻*n)n          |
 *  +-------------------------+-------------------------+--------------------+------------------------------+
 *  | homogeneous operator    | u⁺ = -u⁻                | u⁺ = u⁻            | u⁺ = u⁻ - 2 (u⁻*n)n          |
 *  +-------------------------+-------------------------+--------------------+------------------------------+
 *  | inhomogeneous operator  | u⁺ = -u⁻ + 2g , u⁻ = 0  | u⁺ = u⁻ , u⁻ = 0   | u⁺ = u⁻ - 2 (u⁻*n)n , u⁻ = 0 |
 *  +-------------------------+-------------------------+--------------------+------------------------------+
 *
 */
// clang-format on
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  calculate_interior_value(unsigned int const                       q,
                           FaceIntegrator<dim, dim, Number> const & integrator,
                           OperatorType const &                     operator_type)
{
  // element e⁻
  Tensor<1, dim, VectorizedArray<Number>> value_m;

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    value_m = integrator.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    // do nothing, value_m is already initialized with zeros
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return value_m;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<Number>>
    calculate_exterior_value(Tensor<1, dim, VectorizedArray<Number>> const & value_m,
                             unsigned int const                              q,
                             FaceIntegrator<dim, dim, Number> const &        integrator,
                             OperatorType const &                            operator_type,
                             BoundaryTypeU const &                           boundary_type,
                             types::boundary_id const                        boundary_id,
                             std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor,
                             double const &                                  time)
{
  // element e⁺
  Tensor<1, dim, VectorizedArray<Number>> value_p;

  if(boundary_type == BoundaryTypeU::Dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      auto bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
      auto q_points = integrator.quadrature_point(q);

      auto g = FunctionEvaluator<dim, Number, 1>::value(bc, q_points, time);

      value_p = -value_m + Tensor<1, dim, VectorizedArray<Number>>(2.0 * g);
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
  else if(boundary_type == BoundaryTypeU::Neumann)
  {
    value_p = value_m;
  }
  else if(boundary_type == BoundaryTypeU::Symmetry)
  {
    Tensor<1, dim, VectorizedArray<Number>> normal_m = integrator.get_normal_vector(q);

    value_p = value_m - 2.0 * (value_m * normal_m) * normal_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

/*
 *  This function calculates the exterior velocity on boundary faces
 *  according to:
 *
 *  Dirichlet boundary: u⁺ = -u⁻ + 2g (mirror) or g (direct)
 *  Neumann boundary:   u⁺ = u⁻
 *  symmetry boundary:  u⁺ = u⁻ -(u⁻*n)n - (u⁻*n)n = u⁻ - 2 (u⁻*n)n
 *
 *  The name "nonlinear" indicates that this function is used when
 *  evaluating the nonlinear convective operator.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<Number>>
    calculate_exterior_value_nonlinear(
      Tensor<1, dim, VectorizedArray<Number>> const & u_m,
      unsigned int const                              q,
      FaceIntegrator<dim, dim, Number> &              integrator,
      BoundaryTypeU const &                           boundary_type,
      TypeDirichletBCs const &                        type_dirichlet_bc,
      types::boundary_id const                        boundary_id,
      std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor,
      double const &                                  time)
{
  Tensor<1, dim, VectorizedArray<Number>> u_p;

  if(boundary_type == BoundaryTypeU::Dirichlet)
  {
    auto bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
    auto q_points = integrator.quadrature_point(q);

    auto g = FunctionEvaluator<dim, Number, 1>::value(bc, q_points, time);

    if(type_dirichlet_bc == TypeDirichletBCs::Mirror)
    {
      u_p = -u_m + make_vectorized_array<Number>(2.0) * g;
    }
    else if(type_dirichlet_bc == TypeDirichletBCs::Direct)
    {
      u_p = g;
    }
    else
    {
      AssertThrow(false, ExcMessage("TypeDirichletBCs is not implemented."));
    }
  }
  else if(boundary_type == BoundaryTypeU::Neumann)
  {
    u_p = u_m;
  }
  else if(boundary_type == BoundaryTypeU::Symmetry)
  {
    Tensor<1, dim, VectorizedArray<Number>> normal_m = integrator.get_normal_vector(q);

    u_p = u_m - 2. * (u_m * normal_m) * normal_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return u_p;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<Number>>
    calculate_exterior_value_from_dof_vector(
      Tensor<1, dim, VectorizedArray<Number>> const & value_m,
      unsigned int const                              q,
      FaceIntegrator<dim, dim, Number> const &        integrator_bc,
      OperatorType const &                            operator_type,
      BoundaryTypeU const &                           boundary_type)
{
  // element e⁺
  Tensor<1, dim, VectorizedArray<Number>> value_p;

  if(boundary_type == BoundaryTypeU::Dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      Tensor<1, dim, VectorizedArray<Number>> g = integrator_bc.get_value(q);

      value_p = -value_m + make_vectorized_array<Number>(2.0) * g;
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
  else if(boundary_type == BoundaryTypeU::Neumann)
  {
    value_p = value_m;
  }
  else if(boundary_type == BoundaryTypeU::Symmetry)
  {
    Tensor<1, dim, VectorizedArray<Number>> normal_m = integrator_bc.get_normal_vector(q);

    value_p = value_m - 2.0 * (value_m * normal_m) * normal_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

/*
 * Pressure:
 *
 * These two function calculate the interior/exterior value on boundary faces depending on the
 * operator type, the type of the boundary face and the given boundary conditions.
 *
 *                            +--------------------+----------------------+
 *                            | Neumann boundaries | Dirichlet boundaries |
 *  +-------------------------+--------------------+----------------------+
 *  | full operator           | p⁺ = p⁻            | p⁺ = - p⁻ + 2g       |
 *  +-------------------------+--------------------+----------------------+
 *  | homogeneous operator    | p⁺ = p⁻            | p⁺ = - p⁻            |
 *  +-------------------------+--------------------+----------------------+
 *  | inhomogeneous operator  | p⁺ = 0 , p⁻ = 0    | p⁺ = 2g , p⁻ = 0     |
 *  +-------------------------+--------------------+----------------------+
 *
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_interior_value(unsigned int const                     q,
                           FaceIntegrator<dim, 1, Number> const & integrator,
                           OperatorType const &                   operator_type)
{
  // element e⁻
  VectorizedArray<Number> value_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    value_m = integrator.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    // do nothing, value_m is already initialized with zeros
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return value_m;
}


template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_exterior_value(VectorizedArray<Number> const &                 value_m,
                           unsigned int const                              q,
                           FaceIntegrator<dim, 1, Number> const &          integrator,
                           OperatorType const &                            operator_type,
                           BoundaryTypeP const &                           boundary_type,
                           types::boundary_id const                        boundary_id,
                           std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor,
                           double const &                                  time,
                           double const &                                  inverse_scaling_factor)
{
  VectorizedArray<Number> value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryTypeP::Dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      auto bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
      auto q_points = integrator.quadrature_point(q);

      VectorizedArray<Number> g = FunctionEvaluator<dim, Number, 0>::value(bc, q_points, time);

      value_p = -value_m + 2.0 * inverse_scaling_factor * g;
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
  else if(boundary_type == BoundaryTypeP::Neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  calculate_exterior_value_from_dof_vector(VectorizedArray<Number> const &        value_m,
                                           unsigned int const                     q,
                                           FaceIntegrator<dim, 1, Number> const & integrator_bc,
                                           OperatorType const &                   operator_type,
                                           BoundaryTypeP const &                  boundary_type,
                                           double const & inverse_scaling_factor)
{
  VectorizedArray<Number> value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryTypeP::Dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      VectorizedArray<Number> g = integrator_bc.get_value(q);

      value_p = -value_m + 2.0 * inverse_scaling_factor * g;
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
  else if(boundary_type == BoundaryTypeP::Neumann)
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
 *  These two functions calculates the velocity gradient in normal
 *  direction depending on the operator type, the type of the boundary face
 *  and the given boundary conditions.
 *
 *  Divergence formulation: F(u) = F_nu(u) / nu = ( grad(u) + grad(u)^T )
 *  Laplace formulation:    F(u) = F_nu(u) / nu =  grad(u)
 *
 *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
 *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *  | full operator           | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n + 2h               | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *  | homogeneous operator    | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n                    | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *  | inhomogeneous operator  | F(u⁺)*n = F(u⁻)*n, F(u⁻)*n = 0  | F(u⁺)*n = -F(u⁻)*n + 2h , F(u⁻)*n = 0 | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n, F(u⁻)*n = 0 |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *
 *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
 *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *  | full operator           | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = h                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *  | homogeneous operator    | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = 0                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 *  | inhomogeneous operator  | {{F(u)}}*n = 0                  | {{F(u)}}*n = h                        | {{F(u)}}*n = 0                                     |
 *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
 */
// clang-format on

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, VectorizedArray<Number>>
    calculate_exterior_normal_gradient(
      Tensor<1, dim, VectorizedArray<Number>> const & normal_gradient_m,
      unsigned int const                              q,
      FaceIntegrator<dim, dim, Number> const &        integrator,
      OperatorType const &                            operator_type,
      BoundaryTypeU const &                           boundary_type,
      types::boundary_id const                        boundary_id,
      std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor,
      double const &                                  time,
      bool const                                      variable_normal_vector)
{
  Tensor<1, dim, VectorizedArray<Number>> normal_gradient_p;

  if(boundary_type == BoundaryTypeU::Dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryTypeU::Neumann)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
      auto q_points = integrator.quadrature_point(q);

      Tensor<1, dim, VectorizedArray<Number>> h;
      if(variable_normal_vector == false)
      {
        h = FunctionEvaluator<dim, Number, 1>::value(bc, q_points, time);
      }
      else
      {
        auto normals_m = integrator.get_normal_vector(q);
        h              = FunctionEvaluator<dim, Number, 1>::value(bc, q_points, normals_m, time);
      }

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
  else if(boundary_type == BoundaryTypeU::Symmetry)
  {
    auto normal_m     = integrator.get_normal_vector(q);
    normal_gradient_p = -normal_gradient_m + 2.0 * (normal_gradient_m * normal_m) * normal_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
}

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_ \
        */
