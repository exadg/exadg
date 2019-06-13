/*
 * gradient_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_GRADIENT_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../user_interface/input_parameters.h"
#include "weak_boundary_conditions.h"

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

template<int dim, typename Number>
class GradientOperator
{
public:
  typedef GradientOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorP;

  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorP;

  GradientOperator() : matrix_free(nullptr), eval_time(0.0), inverse_scaling_factor_pressure(1.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &   matrix_free_in,
             GradientOperatorData<dim> const & operator_data_in)
  {
    this->matrix_free   = &matrix_free_in;
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
    matrix_free->loop(&This::cell_loop,
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
    matrix_free->loop(&This::cell_loop,
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

    matrix_free->loop(&This::cell_loop_inhom_operator,
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

    matrix_free->loop(&This::cell_loop,
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

    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_full_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/);
  }

private:
  template<typename CellIntegratorP, typename CellIntegratorU>
  void
  do_cell_integral_weak(CellIntegratorP & pressure, CellIntegratorU & velocity) const
  {
    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      velocity.submit_divergence(-pressure.get_value(q), q);
    }
  }

  template<typename CellIntegratorP, typename CellIntegratorU>
  void
  do_cell_integral_strong(CellIntegratorP & pressure, CellIntegratorU & velocity) const
  {
    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      velocity.submit_value(pressure.get_gradient(q), q);
    }
  }

  template<typename FaceIntegratorP, typename FaceIntegratorU>
  void
  do_face_integral(FaceIntegratorP & pressure_m,
                   FaceIntegratorP & pressure_p,
                   FaceIntegratorU & velocity_m,
                   FaceIntegratorU & velocity_p) const
  {
    for(unsigned int q = 0; q < velocity_m.n_q_points; ++q)
    {
      scalar value_m = pressure_m.get_value(q);
      scalar value_p = pressure_p.get_value(q);

      scalar flux = calculate_flux(value_m, value_p);

      vector flux_times_normal = flux * pressure_m.get_normal_vector(q);

      velocity_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      velocity_p.submit_value(-flux_times_normal, q);
    }
  }

  template<typename FaceIntegratorP, typename FaceIntegratorU>
  void
  do_boundary_integral(FaceIntegratorP &          pressure,
                       FaceIntegratorU &          velocity,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryTypeP boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      scalar flux = make_vectorized_array<Number>(0.0);

      if(operator_data.use_boundary_data == true)
      {
        scalar value_m = calculate_interior_value(q, pressure, operator_type);
        scalar value_p = calculate_exterior_value(value_m,
                                                  q,
                                                  pressure,
                                                  operator_type,
                                                  boundary_type,
                                                  boundary_id,
                                                  operator_data.bc,
                                                  this->eval_time,
                                                  inverse_scaling_factor_pressure);

        flux = calculate_flux(value_m, value_p);
      }
      else // use_boundary_data == false
      {
        scalar value_m = pressure.get_value(q);

        flux = calculate_flux(value_m, value_m /* value_p = value_m */);
      }

      vector flux_times_normal = flux * pressure.get_normal_vector(q);

      velocity.submit_value(flux_times_normal, q);
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

  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    CellIntegratorU velocity(matrix_free,
                             operator_data.dof_index_velocity,
                             operator_data.quad_index);
    CellIntegratorP pressure(matrix_free,
                             operator_data.dof_index_pressure,
                             operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      velocity.reinit(cell);
      pressure.reinit(cell);

      if(operator_data.integration_by_parts == true)
      {
        pressure.gather_evaluate(src, true, false);

        do_cell_integral_weak(pressure, velocity);

        velocity.integrate_scatter(false, true, dst);
      }
      else // integration_by_parts == false
      {
        pressure.gather_evaluate(src, false, true);

        do_cell_integral_strong(pressure, velocity);

        velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity_m(matrix_free,
                                 true,
                                 operator_data.dof_index_velocity,
                                 operator_data.quad_index);
      FaceIntegratorU velocity_p(matrix_free,
                                 false,
                                 operator_data.dof_index_velocity,
                                 operator_data.quad_index);

      FaceIntegratorP pressure_m(matrix_free,
                                 true,
                                 operator_data.dof_index_pressure,
                                 operator_data.quad_index);
      FaceIntegratorP pressure_p(matrix_free,
                                 false,
                                 operator_data.dof_index_pressure,
                                 operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        velocity_m.reinit(face);
        velocity_p.reinit(face);

        pressure_m.reinit(face);
        pressure_p.reinit(face);

        pressure_m.gather_evaluate(src, true, false);
        pressure_p.gather_evaluate(src, true, false);

        do_face_integral(pressure_m, pressure_p, velocity_m, velocity_p);

        velocity_m.integrate_scatter(true, false, dst);
        velocity_p.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity(matrix_free,
                               true,
                               operator_data.dof_index_velocity,
                               operator_data.quad_index);
      FaceIntegratorP pressure(matrix_free,
                               true,
                               operator_data.dof_index_pressure,
                               operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        velocity.reinit(face);
        pressure.reinit(face);

        pressure.gather_evaluate(src, true, false);

        do_boundary_integral(pressure,
                             velocity,
                             OperatorType::homogeneous,
                             matrix_free.get_boundary_id(face));

        velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity(matrix_free,
                               true,
                               operator_data.dof_index_velocity,
                               operator_data.quad_index);
      FaceIntegratorP pressure(matrix_free,
                               true,
                               operator_data.dof_index_pressure,
                               operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        velocity.reinit(face);
        pressure.reinit(face);

        pressure.gather_evaluate(src, true, false);

        do_boundary_integral(pressure,
                             velocity,
                             OperatorType::full,
                             matrix_free.get_boundary_id(face));

        velocity.integrate_scatter(true, false, dst);
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
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    Range const & face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity(matrix_free,
                               true,
                               operator_data.dof_index_velocity,
                               operator_data.quad_index);
      FaceIntegratorP pressure(matrix_free,
                               true,
                               operator_data.dof_index_pressure,
                               operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        velocity.reinit(face);
        pressure.reinit(face);

        do_boundary_integral(pressure,
                             velocity,
                             OperatorType::inhomogeneous,
                             matrix_free.get_boundary_id(face));

        velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

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
