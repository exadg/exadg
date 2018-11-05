/*
 * divergence_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_


#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/operator_type.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct DivergenceOperatorData
{
  DivergenceOperatorData()
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

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class DivergenceOperator
{
public:
  typedef DivergenceOperator<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEEvalVelocity;
  typedef FEEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEEvalPressure;

  typedef FEFaceEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEFaceEvalPressure;

  DivergenceOperator() : data(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &     mf_data,
             DivergenceOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  apply(VectorType & dst, VectorType const & src) const
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
  apply_add(VectorType & dst, VectorType const & src) const
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
    dst = 0.0;
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
      // minus sign due to integration by parts
      fe_eval_pressure.submit_gradient(-fe_eval_velocity.get_value(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_strong(FEEvaluationPressure & fe_eval_pressure,
                          FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      fe_eval_pressure.submit_value(fe_eval_velocity.get_divergence(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_face_integral(FEEvaluationVelocity & fe_eval_velocity_m,
                   FEEvaluationVelocity & fe_eval_velocity_p,
                   FEEvaluationPressure & fe_eval_pressure_m,
                   FEEvaluationPressure & fe_eval_pressure_p) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity_m.n_q_points; ++q)
    {
      vector value_m = fe_eval_velocity_m.get_value(q);
      vector value_p = fe_eval_velocity_p.get_value(q);

      vector flux = calculate_flux(value_m, value_p);

      scalar flux_times_normal = flux * fe_eval_velocity_m.get_normal_vector(q);

      fe_eval_pressure_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      fe_eval_pressure_p.submit_value(-flux_times_normal, q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_boundary_integral(FEEvaluationVelocity &     fe_eval_velocity,
                       FEEvaluationPressure &     fe_eval_pressure,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < fe_eval_pressure.n_q_points; ++q)
    {
      vector flux;

      if(operator_data.use_boundary_data == true)
      {
        vector value_m = calculate_interior_value(q, fe_eval_velocity, operator_type);
        vector value_p = calculate_exterior_value(
          value_m, q, fe_eval_velocity, operator_type, boundary_type, boundary_id);

        flux = calculate_flux(value_m, value_p);
      }
      else // use_boundary_data == false
      {
        vector value_m = fe_eval_velocity.get_value(q);

        flux = calculate_flux(value_m, value_m /* value_p = value_m */);
      }

      scalar flux_times_normal = flux * fe_eval_velocity.get_normal_vector(q);
      fe_eval_pressure.submit_value(flux_times_normal, q);
    }
  }

  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & value_m, vector const & value_p) const
  {
    return 0.5 * (value_m + value_p);
  }

  // clang-format off
  /*
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
  template<typename FEEvaluationVelocity>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_value(unsigned int const           q,
                             FEEvaluationVelocity const & fe_eval_velocity,
                             OperatorType const &         operator_type) const
  {
    // element e⁻
    vector value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_velocity.get_value(q);
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

  template<typename FEEvaluationVelocity>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value(vector const &               value_m,
                             unsigned int const           q,
                             FEEvaluationVelocity const & fe_eval_velocity,
                             OperatorType const &         operator_type,
                             BoundaryTypeU const &        boundary_type,
                             types::boundary_id const     boundary_id = types::boundary_id()) const
  {
    // element e⁺
    vector value_p;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        vector g;

        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
        it                          = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval_velocity.quadrature_point(q);
        evaluate_vectorial_function(g, it->second, q_points, eval_time);

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
      vector normal_m = fe_eval_velocity.get_normal_vector(q);

      value_p = value_m - 2.0 * (value_m * normal_m) * normal_m;
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
      fe_eval_pressure.reinit(cell);

      fe_eval_velocity.reinit(cell);

      if(operator_data.integration_by_parts == true)
      {
        fe_eval_velocity.gather_evaluate(src, true, false, false);

        do_cell_integral_weak(fe_eval_pressure, fe_eval_velocity);

        fe_eval_pressure.integrate_scatter(false, true, dst);
      }
      else // integration_by_parts == false
      {
        fe_eval_velocity.gather_evaluate(src, false, true, false);

        do_cell_integral_strong(fe_eval_pressure, fe_eval_velocity);

        fe_eval_pressure.integrate_scatter(true, false, dst);
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
        fe_eval_pressure.reinit(face);
        fe_eval_pressure_neighbor.reinit(face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity_neighbor.reinit(face);

        fe_eval_velocity.gather_evaluate(src, true, false);
        fe_eval_velocity_neighbor.gather_evaluate(src, true, false);

        do_face_integral(fe_eval_velocity,
                         fe_eval_velocity_neighbor,
                         fe_eval_pressure,
                         fe_eval_pressure_neighbor);

        fe_eval_pressure.integrate_scatter(true, false, dst);
        fe_eval_pressure_neighbor.integrate_scatter(true, false, dst);
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
        fe_eval_pressure.reinit(face);
        fe_eval_velocity.reinit(face);

        fe_eval_velocity.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_velocity,
                             fe_eval_pressure,
                             OperatorType::homogeneous,
                             data.get_boundary_id(face));

        fe_eval_pressure.integrate_scatter(true, false, dst);
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
        fe_eval_pressure.reinit(face);
        fe_eval_velocity.reinit(face);

        fe_eval_velocity.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_velocity,
                             fe_eval_pressure,
                             OperatorType::full,
                             data.get_boundary_id(face));

        fe_eval_pressure.integrate_scatter(true, false, dst);
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
                                    std::pair<unsigned int, unsigned int> const & face_range) const
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
        fe_eval_pressure.reinit(face);
        fe_eval_velocity.reinit(face);

        do_boundary_integral(fe_eval_velocity,
                             fe_eval_pressure,
                             OperatorType::inhomogeneous,
                             data.get_boundary_id(face));

        fe_eval_pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  DivergenceOperatorData<dim> operator_data;

  mutable Number eval_time;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_ \
        */
