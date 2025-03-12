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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/gradient_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
GradientOperator<dim, Number>::GradientOperator()
  : matrix_free(nullptr), time(0.0), inverse_scaling_factor_pressure(1.0), pressure_bc(nullptr)
{
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                          GradientOperatorData<dim> const &       data_in)
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
GradientOperator<dim, Number>::rhs_bc_from_dof_vector(VectorType &       dst,
                                                      VectorType const & pressure) const
{
  pressure_bc = &pressure;

  dst = 0;

  VectorType tmp;
  tmp.reinit(dst, false /* init with 0 */);

  VectorType src_dummy;
  matrix_free->loop(&This::cell_loop_inhom_operator,
                    &This::face_loop_inhom_operator,
                    &This::boundary_face_loop_inhom_operator_bc_from_dof_vector,
                    this,
                    tmp,
                    src_dummy,
                    false /*zero_dst_vector = false*/);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);

  pressure_bc = nullptr;
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::evaluate_bc_from_dof_vector(VectorType &       dst,
                                                           VectorType const & src,
                                                           VectorType const & pressure) const
{
  pressure_bc = &pressure;

  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_full_operator_bc_from_dof_vector,
                    this,
                    dst,
                    src,
                    true /*zero_dst_vector = false*/);

  pressure_bc = nullptr;
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
    if(data.formulation == FormulationPressureGradientTerm::Weak)
    {
      vector flux_times_normal = flux * pressure_m.get_normal_vector(q);

      velocity_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      velocity_p.submit_value(-flux_times_normal, q);
    }
    else if(data.formulation == FormulationPressureGradientTerm::Strong)
    {
      vector normal = pressure_m.get_normal_vector(q);

      velocity_m.submit_value((flux - value_m) * normal, q);
      // minus sign since n⁺ = - n⁻
      velocity_p.submit_value((flux - value_p) * (-normal), q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::do_boundary_integral(
  FaceIntegratorP &                  pressure,
  FaceIntegratorU &                  velocity,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryTypeP boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, pressure, operator_type);

    scalar value_p = dealii::make_vectorized_array<Number>(0.0);
    if(data.use_boundary_data == true)
    {
      value_p = calculate_exterior_value(value_m,
                                         q,
                                         pressure,
                                         operator_type,
                                         boundary_type,
                                         boundary_id,
                                         data.bc,
                                         time,
                                         inverse_scaling_factor_pressure);
    }
    else // use_boundary_data == false
    {
      value_p = value_m;
    }

    scalar flux   = kernel.calculate_flux(value_m, value_p);
    vector normal = pressure.get_normal_vector(q);
    if(data.formulation == FormulationPressureGradientTerm::Weak)
    {
      velocity.submit_value(flux * normal, q);
    }
    else if(data.formulation == FormulationPressureGradientTerm::Strong)
    {
      velocity.submit_value((flux - value_m) * normal, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::do_boundary_integral_from_dof_vector(
  FaceIntegratorP &                  pressure,
  FaceIntegratorP &                  pressure_bc,
  FaceIntegratorU &                  velocity,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryTypeP boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, pressure, operator_type);

    scalar value_p = dealii::make_vectorized_array<Number>(0.0);
    if(data.use_boundary_data == true)
    {
      value_p = calculate_exterior_value_from_dof_vector(
        value_m, q, pressure_bc, operator_type, boundary_type, inverse_scaling_factor_pressure);
    }
    else // use_boundary_data == false
    {
      value_p = value_m;
    }

    scalar flux   = kernel.calculate_flux(value_m, value_p);
    vector normal = pressure.get_normal_vector(q);
    if(data.formulation == FormulationPressureGradientTerm::Weak)
    {
      velocity.submit_value(flux * normal, q);
    }
    else if(data.formulation == FormulationPressureGradientTerm::Strong)
    {
      velocity.submit_value((flux - value_m) * normal, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                         VectorType &                            dst,
                                         VectorType const &                      src,
                                         Range const &                           cell_range) const
{
  CellIntegratorU velocity(matrix_free, data.dof_index_velocity, data.quad_index);
  CellIntegratorP pressure(matrix_free, data.dof_index_pressure, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    velocity.reinit(cell);
    pressure.reinit(cell);

    if(data.integration_by_parts == true and
       data.formulation == FormulationPressureGradientTerm::Weak)
    {
      pressure.gather_evaluate(src, dealii::EvaluationFlags::values);

      do_cell_integral_weak(pressure, velocity);

      velocity.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
    }
    else
    {
      pressure.gather_evaluate(src, dealii::EvaluationFlags::gradients);

      do_cell_integral_strong(pressure, velocity);

      velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::face_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                         VectorType &                            dst,
                                         VectorType const &                      src,
                                         Range const &                           face_range) const
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

      pressure_m.gather_evaluate(src, dealii::EvaluationFlags::values);
      pressure_p.gather_evaluate(src, dealii::EvaluationFlags::values);

      do_face_integral(pressure_m, pressure_p, velocity_m, velocity_p);

      velocity_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      velocity_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_hom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity.reinit(face);
      pressure.reinit(face);

      pressure.gather_evaluate(src, dealii::EvaluationFlags::values);

      do_boundary_integral(pressure,
                           velocity,
                           OperatorType::homogeneous,
                           matrix_free.get_boundary_id(face));

      velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_full_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity.reinit(face);
      pressure.reinit(face);

      pressure.gather_evaluate(src, dealii::EvaluationFlags::values);

      do_boundary_integral(pressure,
                           velocity,
                           OperatorType::full,
                           matrix_free.get_boundary_id(face));

      velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::cell_loop_inhom_operator(dealii::MatrixFree<dim, Number> const &,
                                                        VectorType &,
                                                        VectorType const &,
                                                        Range const &) const
{
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const &,
                                                        VectorType &,
                                                        VectorType const &,
                                                        Range const &) const
{
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  (void)src;

  if(data.integration_by_parts == true and data.use_boundary_data == true)
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

      velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_inhom_operator_bc_from_dof_vector(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  if(data.integration_by_parts == true and data.use_boundary_data == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);
    FaceIntegratorP pressure_ext(matrix_free, true, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity.reinit(face);

      pressure.reinit(face);

      pressure_ext.reinit(face);
      pressure_ext.gather_evaluate(*pressure_bc, dealii::EvaluationFlags::values);

      do_boundary_integral_from_dof_vector(pressure,
                                           pressure_ext,
                                           velocity,
                                           OperatorType::inhomogeneous,
                                           matrix_free.get_boundary_id(face));

      velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template<int dim, typename Number>
void
GradientOperator<dim, Number>::boundary_face_loop_full_operator_bc_from_dof_vector(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if(data.integration_by_parts == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);
    FaceIntegratorP pressure_ext(matrix_free, true, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity.reinit(face);

      pressure.reinit(face);
      pressure.gather_evaluate(src, dealii::EvaluationFlags::values);

      pressure_ext.reinit(face);
      pressure_ext.gather_evaluate(*pressure_bc, dealii::EvaluationFlags::values);

      do_boundary_integral_from_dof_vector(
        pressure, pressure_ext, velocity, OperatorType::full, matrix_free.get_boundary_id(face));

      velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template class GradientOperator<2, float>;
template class GradientOperator<2, double>;

template class GradientOperator<3, float>;
template class GradientOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
