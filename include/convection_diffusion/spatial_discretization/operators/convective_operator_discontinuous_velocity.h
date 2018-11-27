/*
 * convective_operator_discontinuous_velocity.h
 *
 *  Created on: Nov 26, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_DISCONTINUOUS_VELOCITY_H_
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_DISCONTINUOUS_VELOCITY_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

#include "operators/operator_type.h"

namespace ConvDiff
{
template<int dim>
struct ConvectiveOperatorDisVelData
{
  ConvectiveOperatorDisVelData()
    : dof_index(0),
      dof_index_velocity(1),
      quad_index(0),
      numerical_flux_formulation(NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
  {
  }

  unsigned int dof_index;
  unsigned int dof_index_velocity;
  unsigned int quad_index;

  NumericalFluxConvectiveOperator numerical_flux_formulation;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, int degree, int degree_velocity, typename Number>
class ConvectiveOperatorDisVel
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef ConvectiveOperatorDisVel<dim, degree, degree_velocity, Number> This;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>            FEEvalCell;
  typedef FEEvaluation<dim, degree_velocity, degree + 1, dim, Number> FEEvalCellVelocity;

  typedef FEFaceEvaluation<dim, degree, degree + 1, 1, Number>            FEEvalFace;
  typedef FEFaceEvaluation<dim, degree_velocity, degree + 1, dim, Number> FEEvalFaceVelocity;

  ConvectiveOperatorDisVel() : data(nullptr), evaluation_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &           data_in,
             ConvectiveOperatorDisVelData<dim> const & operator_data_in)
  {
    this->data          = &data_in;
    this->operator_data = operator_data_in;

    data->initialize_dof_vector(velocity, operator_data.dof_index_velocity);
  }

  // apply matrix vector multiplication (homogeneous operator)
  void
  apply(VectorType & dst, VectorType const & src, VectorType const & vector) const
  {
    dst = 0;
    apply_add(dst, src, vector);
  }

  void
  apply_add(VectorType & dst, VectorType const & src, VectorType const & vector) const
  {
    set_velocity(vector);

    data->loop(
      &This::cell_loop, &This::face_loop, &This::boundary_face_loop_hom_operator, this, dst, src);
  }

  // evaluate operator (full operator), the velocity has to be set independently
  void
  evaluate(VectorType & dst, VectorType const & src, double const time) const
  {
    dst = 0;
    evaluate_add(dst, src, time);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, double const time) const
  {
    evaluation_time = time;

    data->loop(
      &This::cell_loop, &This::face_loop, &This::boundary_face_loop_full_operator, this, dst, src);
  }

  // calculate rhs (inhomogeneous operator), the velocity has to be set independently
  void
  rhs(VectorType & dst, double const time) const
  {
    dst = 0;
    rhs_add(dst, time);
  }

  void
  rhs_add(VectorType & dst, double const time) const
  {
    evaluation_time = time;

    VectorType tmp;
    tmp.reinit(dst, false);

    data->loop(&This::cell_loop_empty,
               &This::face_loop_empty,
               &This::boundary_face_loop_inhom_operator,
               this,
               tmp,
               tmp);

    // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
    dst.add(-1.0, tmp);
  }

  void
  set_velocity(VectorType const & velocity_in) const
  {
    velocity = velocity_in;

    velocity.update_ghost_values();
  }

private:
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
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_interior_value(unsigned int const   q,
                             FEEvalFace const &   fe_eval,
                             OperatorType const & operator_type) const
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
      return fe_eval.get_value(q);
    else if(operator_type == OperatorType::inhomogeneous)
      return make_vectorized_array<Number>(0.0);
    else
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));

    return make_vectorized_array<Number>(0.0);
  }

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_exterior_value(scalar const &           value_m,
                             unsigned int const       q,
                             FEEvalFace const &       fe_eval,
                             OperatorType const &     operator_type,
                             BoundaryType const &     boundary_type,
                             types::boundary_id const boundary_id) const
  {
    scalar value_p = make_vectorized_array<Number>(0.0);

    if(boundary_type == BoundaryType::dirichlet)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

        scalar g = evaluate_scalar_function(it->second, q_points, evaluation_time);

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

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_central_flux(scalar const & value_m,
                           scalar const & value_p,
                           scalar const & normal_velocity_m,
                           scalar const & normal_velocity_p) const
  {
    return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p);
  }

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_lax_friedrichs_flux(scalar const & value_m,
                                  scalar const & value_p,
                                  scalar const & normal_velocity_m,
                                  scalar const & normal_velocity_p) const
  {
    scalar jump_value = value_m - value_p;
    scalar lambda     = std::max(std::abs(normal_velocity_m), std::abs(normal_velocity_p));

    return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p) +
           0.5 * lambda * jump_value;
  }

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_flux(scalar const & value_m,
                   scalar const & value_p,
                   scalar const & normal_velocity_m,
                   scalar const & normal_velocity_p) const
  {
    scalar flux = make_vectorized_array<Number>(0.0);

    if(operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
    {
      flux = calculate_central_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
    }
    else if(operator_data.numerical_flux_formulation ==
            NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
    {
      flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
    }

    return flux;
  }

  void
  do_cell_integral(FEEvalCell & fe_eval, FEEvalCellVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_gradient(-fe_eval.get_value(q) * fe_eval_velocity.get_value(q), q);
    }
  }

  void
  do_face_integral(FEEvalFace &         fe_eval,
                   FEEvalFace &         fe_eval_neighbor,
                   FEEvalFaceVelocity & fe_eval_velocity,
                   FEEvalFaceVelocity & fe_eval_velocity_neighbor) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar value_m = fe_eval.get_value(q);
      scalar value_p = fe_eval_neighbor.get_value(q);

      vector velocity_m = fe_eval_velocity.get_value(q);
      vector velocity_p = fe_eval_velocity_neighbor.get_value(q);

      vector normal = fe_eval.get_normal_vector(q);

      scalar normal_velocity_m = velocity_m * normal;
      scalar normal_velocity_p = velocity_p * normal;

      scalar flux = calculate_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);

      fe_eval.submit_value(flux, q);
      fe_eval_neighbor.submit_value(-flux, q);
    }
  }

  void
  do_boundary_integral(FEEvalFace &               fe_eval,
                       FEEvalFaceVelocity &       fe_eval_velocity,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar value_m = calculate_interior_value(q, fe_eval, operator_type);
      scalar value_p =
        calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);

      vector velocity_m = fe_eval_velocity.get_value(q);

      vector normal = fe_eval.get_normal_vector(q);

      scalar normal_velocity_m = velocity_m * normal;

      scalar flux = calculate_flux(value_m, value_p, normal_velocity_m, normal_velocity_m);

      fe_eval.submit_value(flux, q);
    }
  }

  /*
   *  Calculate cell integrals.
   */
  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellVelocity fe_eval_velocity(data,
                                        operator_data.dof_index_velocity,
                                        operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, true, false, false);

      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.gather_evaluate(velocity, true, false, false);

      do_cell_integral(fe_eval, fe_eval_velocity);

      fe_eval.integrate_scatter(false, true, dst);
    }
  }

  /*
   *  Calculate interior face integrals.
   */
  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFace fe_eval_neighbor(data, false, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceVelocity fe_eval_velocity(data,
                                        true,
                                        operator_data.dof_index_velocity,
                                        operator_data.quad_index);

    FEEvalFaceVelocity fe_eval_velocity_neighbor(data,
                                                 false,
                                                 operator_data.dof_index_velocity,
                                                 operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.gather_evaluate(src, true, false);

      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.gather_evaluate(velocity, true, false);

      fe_eval_velocity_neighbor.reinit(face);
      fe_eval_velocity_neighbor.gather_evaluate(velocity, true, false);

      do_face_integral(fe_eval, fe_eval_neighbor, fe_eval_velocity, fe_eval_velocity_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  /*
   *  Calculate boundary face integrals.
   */
  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & data,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceVelocity fe_eval_velocity(data,
                                        true,
                                        operator_data.dof_index_velocity,
                                        operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.gather_evaluate(src, true, false);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.gather_evaluate(velocity, true, false);

      types::boundary_id boundary_id = data.get_boundary_id(face);
      do_boundary_integral(fe_eval, fe_eval_velocity, OperatorType::homogeneous, boundary_id);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & data,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceVelocity fe_eval_velocity(data,
                                        true,
                                        operator_data.dof_index_velocity,
                                        operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.gather_evaluate(src, true, false);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.gather_evaluate(velocity, true, false);

      types::boundary_id boundary_id = data.get_boundary_id(face);
      do_boundary_integral(fe_eval, fe_eval_velocity, OperatorType::full, boundary_id);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & data,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceVelocity fe_eval_velocity(data,
                                        true,
                                        operator_data.dof_index_velocity,
                                        operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.gather_evaluate(velocity, true, false);

      types::boundary_id boundary_id = data.get_boundary_id(face);
      do_boundary_integral(fe_eval, fe_eval_velocity, OperatorType::inhomogeneous, boundary_id);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  void
  cell_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  face_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  MatrixFree<dim, Number> const * data;

  ConvectiveOperatorDisVelData<dim> operator_data;

  mutable VectorType velocity;

  mutable double evaluation_time;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_DISCONTINUOUS_VELOCITY_H_ \
        */
