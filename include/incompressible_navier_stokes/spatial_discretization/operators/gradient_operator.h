/*
 * gradient_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/operator_type.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct GradientOperatorData
{
  GradientOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      quad_index(0),
      integration_by_parts(true),
      use_boundary_data(true)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  bool integration_by_parts;
  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorP<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class GradientOperator
{
public:
  typedef GradientOperator<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEEvalVelocity;
  typedef FEEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEEvalPressure;

  typedef FEFaceEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEFaceEvalPressure;

  GradientOperator() : data(nullptr), eval_time(0.0), inverse_scaling_factor_pressure(1.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &   mf_data,
             GradientOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  set_scaling_factor_pressure(Number const & scaling_factor)
  {
    inverse_scaling_factor_pressure = 1.0 / scaling_factor;
  }

  void
  apply(VectorType & dst, const VectorType & src) const
  {
    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_hom_operator,
               this,
               dst,
               src,
               true /*zero_dst_vector = true*/);
  }

  void
  apply_add(VectorType & dst, const VectorType & src) const
  {
    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_hom_operator,
               this,
               dst,
               src,
               false /*zero_dst_vector = false*/);
  }

  void
  rhs(VectorType & dst, Number const evaluation_time) const
  {
    dst = 0;
    rhs_add(dst, evaluation_time);
  }

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType tmp;
    tmp.reinit(dst, false /* init with 0 */);

    data->loop(&This::cell_loop_inhom_operator,
               &This::face_loop_inhom_operator,
               &This::boundary_face_loop_inhom_operator,
               this,
               tmp,
               tmp,
               false /*zero_dst_vector = false*/);

    // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
    dst.add(-1.0, tmp);
  }

  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_full_operator,
               this,
               dst,
               src,
               true /*zero_dst_vector = true*/);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_full_operator,
               this,
               dst,
               src,
               false /*zero_dst_vector = false*/);
  }

private:
  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_weak(FEEvaluationPressure & fe_eval_pressure,
                        FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      fe_eval_velocity.submit_divergence(-fe_eval_pressure.get_value(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_strong(FEEvaluationPressure & fe_eval_pressure,
                          FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      fe_eval_velocity.submit_value(fe_eval_pressure.get_gradient(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_face_integral(FEEvaluationPressure & fe_eval_pressure_m,
                   FEEvaluationPressure & fe_eval_pressure_p,
                   FEEvaluationVelocity & fe_eval_velocity_m,
                   FEEvaluationVelocity & fe_eval_velocity_p) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity_m.n_q_points; ++q)
    {
      scalar value_m = fe_eval_pressure_m.get_value(q);
      scalar value_p = fe_eval_pressure_p.get_value(q);

      scalar flux = calculate_flux(value_m, value_p);

      vector flux_times_normal = flux * fe_eval_pressure_m.get_normal_vector(q);

      fe_eval_velocity_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      fe_eval_velocity_p.submit_value(-flux_times_normal, q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_boundary_integral(FEEvaluationPressure &     fe_eval_pressure,
                       FEEvaluationVelocity &     fe_eval_velocity,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryTypeP boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      scalar flux = make_vectorized_array<Number>(0.0);

      if(operator_data.use_boundary_data == true)
      {
        scalar value_m = calculate_interior_value(q, fe_eval_pressure, operator_type);

        scalar value_p = calculate_exterior_value(
          value_m, q, fe_eval_pressure, operator_type, boundary_type, boundary_id);

        flux = calculate_flux(value_m, value_p);
      }
      else // use_boundary_data == false
      {
        scalar value_m = fe_eval_pressure.get_value(q);

        flux = calculate_flux(value_m, value_m /* value_p = value_m */);
      }

      vector flux_times_normal = flux * fe_eval_pressure.get_normal_vector(q);

      fe_eval_velocity.submit_value(flux_times_normal, q);
    }
  }


  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux(scalar const & value_m, scalar const & value_p) const
  {
    return 0.5 * (value_m + value_p);
  }

  /*
   *  These two function calculate the interior/exterior value on boundary faces depending on the
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
  template<typename FEEvaluationPressure>
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_interior_value(unsigned int const           q,
                             FEEvaluationPressure const & fe_eval_pressure,
                             OperatorType const &         operator_type) const
  {
    // element e⁻
    scalar value_m = make_vectorized_array<Number>(0.0);

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_pressure.get_value(q);
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


  template<typename FEEvaluationPressure>
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_exterior_value(scalar const &               value_m,
                             unsigned int const           q,
                             FEEvaluationPressure const & fe_eval_pressure,
                             OperatorType const &         operator_type,
                             BoundaryTypeP const &        boundary_type,
                             types::boundary_id const     boundary_id = types::boundary_id()) const
  {
    scalar value_p = make_vectorized_array<Number>(0.0);

    if(boundary_type == BoundaryTypeP::Dirichlet)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval_pressure.quadrature_point(q);

        scalar g = evaluate_scalar_function(it->second, q_points, eval_time);

        value_p = -value_m + 2.0 * inverse_scaling_factor_pressure * g;
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

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEvalVelocity fe_eval_velocity(data,
                                    operator_data.dof_index_velocity,
                                    operator_data.quad_index);
    FEEvalPressure fe_eval_pressure(data,
                                    operator_data.dof_index_pressure,
                                    operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_pressure.reinit(cell);

      if(operator_data.integration_by_parts == true)
      {
        fe_eval_pressure.gather_evaluate(src, true, false);

        do_cell_integral_weak(fe_eval_pressure, fe_eval_velocity);

        fe_eval_velocity.integrate_scatter(false, true, dst);
      }
      else // integration_by_parts == false
      {
        fe_eval_pressure.gather_evaluate(src, false, true);

        do_cell_integral_strong(fe_eval_pressure, fe_eval_velocity);

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalVelocity fe_eval_velocity_neighbor(data,
                                                   false,
                                                   operator_data.dof_index_velocity,
                                                   operator_data.quad_index);

      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure_neighbor(data,
                                                   false,
                                                   operator_data.dof_index_pressure,
                                                   operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_velocity_neighbor.reinit(face);

        fe_eval_pressure.reinit(face);
        fe_eval_pressure_neighbor.reinit(face);

        fe_eval_pressure.gather_evaluate(src, true, false);
        fe_eval_pressure_neighbor.gather_evaluate(src, true, false);

        do_face_integral(fe_eval_pressure,
                         fe_eval_pressure_neighbor,
                         fe_eval_velocity,
                         fe_eval_velocity_neighbor);

        fe_eval_velocity.integrate_scatter(true, false, dst);
        fe_eval_velocity_neighbor.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & data,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_pressure.reinit(face);

        fe_eval_pressure.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_pressure,
                             fe_eval_velocity,
                             OperatorType::homogeneous,
                             data.get_boundary_id(face));

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & data,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_pressure.reinit(face);

        fe_eval_pressure.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_pressure,
                             fe_eval_velocity,
                             OperatorType::full,
                             data.get_boundary_id(face));

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  cell_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  face_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & data,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    Range const & face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_pressure.reinit(face);

        do_boundary_integral(fe_eval_pressure,
                             fe_eval_velocity,
                             OperatorType::inhomogeneous,
                             data.get_boundary_id(face));

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  GradientOperatorData<dim> operator_data;

  mutable Number eval_time;

  // if the continuity equation of the incompressible Navier-Stokes
  // equations is scaled by a constant factor, the system of equations
  // is solved for a modified pressure p^* = 1/scaling_factor * p. Hence,
  // when applying the gradient operator to this modified pressure we have
  // to make sure that we also apply the correct boundary conditions for p^*,
  // i.e., g_p^* = 1/scaling_factor * g_p
  Number inverse_scaling_factor_pressure;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_ \
        */
