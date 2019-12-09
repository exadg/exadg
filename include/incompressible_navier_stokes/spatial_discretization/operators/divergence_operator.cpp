/*
 * divergence_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include "divergence_operator.h"

namespace IncNS
{
template<int dim, typename Number>
DivergenceOperator<dim, Number>::DivergenceOperator()
  : matrix_free(nullptr), time(0.0), velocity_bc(nullptr)
{
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free_in,
                                        DivergenceOperatorData<dim> const & data_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;
}

template<int dim, typename Number>
DivergenceOperatorData<dim> const &
DivergenceOperator<dim, Number>::get_operator_data() const
{
  return this->data;
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
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
DivergenceOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
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
DivergenceOperator<dim, Number>::rhs(VectorType & dst, Number const evaluation_time) const
{
  dst = 0.0;
  rhs_add(dst, evaluation_time);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::rhs_bc_from_dof_vector(VectorType &       dst,
                                                        VectorType const & velocity) const
{
  velocity_bc = &velocity;

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

  velocity_bc = nullptr;
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::rhs_add(VectorType & dst, Number const evaluation_time) const
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
DivergenceOperator<dim, Number>::evaluate(VectorType &       dst,
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
DivergenceOperator<dim, Number>::evaluate_add(VectorType &       dst,
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
DivergenceOperator<dim, Number>::do_cell_integral_weak(CellIntegratorP & pressure,
                                                       CellIntegratorU & velocity) const
{
  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    pressure.submit_gradient(kernel.get_volume_flux_weak(velocity, q), q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_cell_integral_strong(CellIntegratorP & pressure,
                                                         CellIntegratorU & velocity) const
{
  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    pressure.submit_value(kernel.get_volume_flux_strong(velocity, q), q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_face_integral(FaceIntegratorU & velocity_m,
                                                  FaceIntegratorU & velocity_p,
                                                  FaceIntegratorP & pressure_m,
                                                  FaceIntegratorP & pressure_p) const
{
  for(unsigned int q = 0; q < velocity_m.n_q_points; ++q)
  {
    vector value_m = velocity_m.get_value(q);
    vector value_p = velocity_p.get_value(q);

    vector flux;
    if(data.formulation == FormulationVelocityDivergenceTerm::Weak)
      flux = kernel.calculate_flux_weak(value_m, value_p);
    else if(data.formulation == FormulationVelocityDivergenceTerm::Strong)
      flux = kernel.calculate_flux_strong(value_m, value_p);
    else
      AssertThrow(false, ExcMessage("Not implemented."));

    scalar flux_times_normal = flux * velocity_m.get_normal_vector(q);

    pressure_m.submit_value(flux_times_normal, q);
    // minus sign since n⁺ = - n⁻
    pressure_p.submit_value(-flux_times_normal, q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_boundary_integral(FaceIntegratorU &          velocity,
                                                      FaceIntegratorP &          pressure,
                                                      OperatorType const &       operator_type,
                                                      types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < pressure.n_q_points; ++q)
  {
    vector flux;

    vector value_m = calculate_interior_value(q, velocity, operator_type);
    vector value_p;
    if(data.use_boundary_data == true)
    {
      value_p = calculate_exterior_value(
        value_m, q, velocity, operator_type, boundary_type, boundary_id, data.bc, time);
    }
    else // use_boundary_data == false
    {
      value_p = value_m;
    }

    if(data.formulation == FormulationVelocityDivergenceTerm::Weak)
      flux = kernel.calculate_flux_weak(value_m, value_p);
    else if(data.formulation == FormulationVelocityDivergenceTerm::Strong)
      flux = kernel.calculate_flux_strong(value_m, value_p);
    else
      AssertThrow(false, ExcMessage("Not implemented."));

    scalar flux_times_normal = flux * velocity.get_normal_vector(q);
    pressure.submit_value(flux_times_normal, q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_boundary_integral_from_dof_vector(
  FaceIntegratorU &          velocity,
  FaceIntegratorU &          velocity_bc,
  FaceIntegratorP &          pressure,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < pressure.n_q_points; ++q)
  {
    vector flux;

    vector value_m = calculate_interior_value(q, velocity, operator_type);
    vector value_p;
    if(data.use_boundary_data == true)
    {
      value_p = calculate_exterior_value_from_dof_vector(
        value_m, q, velocity_bc, operator_type, boundary_type);
    }
    else // use_boundary_data == false
    {
      value_p = value_m;
    }

    if(data.formulation == FormulationVelocityDivergenceTerm::Weak)
      flux = kernel.calculate_flux_weak(value_m, value_p);
    else if(data.formulation == FormulationVelocityDivergenceTerm::Strong)
      flux = kernel.calculate_flux_strong(value_m, value_p);
    else
      AssertThrow(false, ExcMessage("Not implemented."));

    scalar flux_times_normal = flux * velocity.get_normal_vector(q);
    pressure.submit_value(flux_times_normal, q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & matrix_free,
                                           VectorType &                    dst,
                                           VectorType const &              src,
                                           Range const &                   cell_range) const
{
  CellIntegratorU velocity(matrix_free, data.dof_index_velocity, data.quad_index);
  CellIntegratorP pressure(matrix_free, data.dof_index_pressure, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    pressure.reinit(cell);

    velocity.reinit(cell);

    if(data.integration_by_parts == true &&
       data.formulation == FormulationVelocityDivergenceTerm::Weak)
    {
      velocity.gather_evaluate(src, true, false, false);

      do_cell_integral_weak(pressure, velocity);

      pressure.integrate_scatter(false, true, dst);
    }
    else // integration_by_parts == false
    {
      velocity.gather_evaluate(src, false, true, false);

      do_cell_integral_strong(pressure, velocity);

      pressure.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::face_loop(MatrixFree<dim, Number> const & matrix_free,
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
      pressure_m.reinit(face);
      pressure_p.reinit(face);

      velocity_m.reinit(face);
      velocity_p.reinit(face);

      velocity_m.gather_evaluate(src, true, false);
      velocity_p.gather_evaluate(src, true, false);

      do_face_integral(velocity_m, velocity_p, pressure_m, pressure_p);

      pressure_m.integrate_scatter(true, false, dst);
      pressure_p.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::boundary_face_loop_hom_operator(
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
      pressure.reinit(face);
      velocity.reinit(face);

      velocity.gather_evaluate(src, true, false);

      do_boundary_integral(velocity,
                           pressure,
                           OperatorType::homogeneous,
                           matrix_free.get_boundary_id(face));

      pressure.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::boundary_face_loop_full_operator(
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
      pressure.reinit(face);
      velocity.reinit(face);

      velocity.gather_evaluate(src, true, false);

      do_boundary_integral(velocity,
                           pressure,
                           OperatorType::full,
                           matrix_free.get_boundary_id(face));

      pressure.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::cell_loop_inhom_operator(MatrixFree<dim, Number> const &,
                                                          VectorType &,
                                                          VectorType const &,
                                                          Range const &) const
{
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::face_loop_inhom_operator(MatrixFree<dim, Number> const &,
                                                          VectorType &,
                                                          VectorType const &,
                                                          Range const &) const
{
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::boundary_face_loop_inhom_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  if(data.integration_by_parts == true && data.use_boundary_data == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);

    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      pressure.reinit(face);
      velocity.reinit(face);

      do_boundary_integral(velocity,
                           pressure,
                           OperatorType::inhomogeneous,
                           matrix_free.get_boundary_id(face));

      pressure.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::boundary_face_loop_inhom_operator_bc_from_dof_vector(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &,
  Range const & face_range) const
{
  if(data.integration_by_parts == true && data.use_boundary_data == true)
  {
    FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorU velocity_ext(matrix_free, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity.reinit(face);

      velocity_ext.reinit(face);
      velocity_ext.gather_evaluate(*velocity_bc, true, false);

      pressure.reinit(face);

      do_boundary_integral_from_dof_vector(velocity,
                                           velocity_ext,
                                           pressure,
                                           OperatorType::inhomogeneous,
                                           matrix_free.get_boundary_id(face));

      pressure.integrate_scatter(true, false, dst);
    }
  }
}


template class DivergenceOperator<2, float>;
template class DivergenceOperator<2, double>;

template class DivergenceOperator<3, float>;
template class DivergenceOperator<3, double>;

} // namespace IncNS
