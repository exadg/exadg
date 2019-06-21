/*
 * gradient_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "gradient_operator.h"

namespace IncNS
{
template<int dim, typename Number>
GradientOperator<dim, Number>::GradientOperator()
  : matrix_free(nullptr), time(0.0), inverse_scaling_factor_pressure(1.0)
{
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &   matrix_free_in,
                                      GradientOperatorData<dim> const & data_in)
{
  matrix_free = &matrix_free_in;
  data        = data_in;
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::set_scaling_factor_pressure(double const & scaling_factor)
{
  inverse_scaling_factor_pressure = 1.0 / scaling_factor;
}

template<int dim, typename Number>
GradientOperatorData<dim> const &
GradientOperator<dim, Number>::get_operator_data() const
{
  return this->data;
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_hom_operator,
                    this,
                    dst,
                    src,
                    true /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_hom_operator,
                    this,
                    dst,
                    src,
                    false /*zero_dst_vector = false*/);
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::rhs(VectorType & dst, Number const evaluation_time) const
{
  dst = 0;
  rhs_add(dst, evaluation_time);
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::rhs_add(VectorType & dst, Number const evaluation_time) const
{
  time = evaluation_time;

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

template<int dim, typename Number>
void
GradientOperator<dim, Number>::evaluate(VectorType &       dst,
                                        VectorType const & src,
                                        Number const       evaluation_time) const
{
  time = evaluation_time;

  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_full_operator,
                    this,
                    dst,
                    src,
                    true /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::evaluate_add(VectorType &       dst,
                                            VectorType const & src,
                                            Number const       evaluation_time) const
{
  time = evaluation_time;

  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_full_operator,
                    this,
                    dst,
                    src,
                    false /*zero_dst_vector = false*/);
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::do_cell_integral_weak(CellIntegratorP & pressure,
                                                     CellIntegratorU & velocity) const
{
  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    velocity.submit_divergence(kernel.get_volume_flux_weak(pressure, q), q);
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::do_cell_integral_strong(CellIntegratorP & pressure,
                                                       CellIntegratorU & velocity) const
{
  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    velocity.submit_value(kernel.get_volume_flux_strong(pressure, q), q);
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::do_face_integral(FaceIntegratorP & pressure_m,
                                                FaceIntegratorP & pressure_p,
                                                FaceIntegratorU & velocity_m,
                                                FaceIntegratorU & velocity_p) const
{
  for(unsigned int q = 0; q < velocity_m.n_q_points; ++q)
  {
    scalar value_m = pressure_m.get_value(q);
    scalar value_p = pressure_p.get_value(q);

    scalar flux = kernel.calculate_flux(value_m, value_p);

    vector flux_times_normal = flux * pressure_m.get_normal_vector(q);

    velocity_m.submit_value(flux_times_normal, q);
    // minus sign since n⁺ = - n⁻
    velocity_p.submit_value(-flux_times_normal, q);
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::do_boundary_integral(FaceIntegratorP &          pressure,
                                                    FaceIntegratorU &          velocity,
                                                    OperatorType const &       operator_type,
                                                    types::boundary_id const & boundary_id) const
{
  BoundaryTypeP boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    scalar flux = make_vectorized_array<Number>(0.0);

    if(data.use_boundary_data == true)
    {
      scalar value_m = calculate_interior_value(q, pressure, operator_type);
      scalar value_p = calculate_exterior_value(value_m,
                                                q,
                                                pressure,
                                                operator_type,
                                                boundary_type,
                                                boundary_id,
                                                data.bc,
                                                time,
                                                inverse_scaling_factor_pressure);

      flux = kernel.calculate_flux(value_m, value_p);
    }
    else // use_boundary_data == false
    {
      scalar value_m = pressure.get_value(q);

      flux = kernel.calculate_flux(value_m, value_m /* value_p = value_m */);
    }

    vector flux_times_normal = flux * pressure.get_normal_vector(q);

    velocity.submit_value(flux_times_normal, q);
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & matrix_free,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   cell_range) const
{
  CellIntegratorU velocity(matrix_free, data.dof_index_velocity, data.quad_index);
  CellIntegratorP pressure(matrix_free, data.dof_index_pressure, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    velocity.reinit(cell);
    pressure.reinit(cell);

    if(data.integration_by_parts == true)
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

template<int dim, typename Number>
void
GradientOperator<dim, Number>::face_loop(MatrixFree<dim, Number> const & matrix_free,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   face_range) const
{
  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity_m(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorU velocity_p(matrix_free, false, data.dof_index_velocity, data.quad_index);

    FaceIntegratorP pressure_m(matrix_free, true, data.dof_index_pressure, data.quad_index);
    FaceIntegratorP pressure_p(matrix_free, false, data.dof_index_pressure, data.quad_index);

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

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_hom_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

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

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_full_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

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

template<int dim, typename Number>
void
GradientOperator<dim, Number>::cell_loop_inhom_operator(MatrixFree<dim, Number> const &,
                                                        VectorType &,
                                                        VectorType const &,
                                                        Range const &) const
{
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::face_loop_inhom_operator(MatrixFree<dim, Number> const &,
                                                        VectorType &,
                                                        VectorType const &,
                                                        Range const &) const
{
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_inhom_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  (void)src;

  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

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

template class GradientOperator<2, float>;
template class GradientOperator<2, double>;

template class GradientOperator<3, float>;
template class GradientOperator<3, double>;

} // namespace IncNS
