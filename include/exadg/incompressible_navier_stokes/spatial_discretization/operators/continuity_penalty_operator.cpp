/*
 * continuity_penalty_operator.cpp
 *
 *  Created on: Jun 25, 2019
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/continuity_penalty_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
ContinuityPenaltyOperator<dim, Number>::ContinuityPenaltyOperator()
  : matrix_free(nullptr), time(0.0)
{
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  ContinuityPenaltyData<dim> const &      data,
  std::shared_ptr<Kernel> const           kernel)
{
  this->matrix_free = &matrix_free;
  this->data        = data;
  this->kernel      = kernel;
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::update(VectorType const & velocity)
{
  kernel->calculate_penalty_parameter(velocity);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop,
                    &This::boundary_face_loop_hom,
                    this,
                    dst,
                    src,
                    true,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop,
                    &This::boundary_face_loop_hom,
                    this,
                    dst,
                    src,
                    false,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::rhs(VectorType & dst, Number const evaluation_time) const
{
  dst = 0.0;
  rhs_add(dst, evaluation_time);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::rhs_add(VectorType & dst,
                                                Number const evaluation_time) const
{
  time = evaluation_time;

  VectorType tmp;
  tmp.reinit(dst, false /* init with 0 */);

  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::boundary_face_loop_inhom,
                    this,
                    tmp,
                    tmp,
                    false /*zero_dst_vector = false*/);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::evaluate(VectorType &       dst,
                                                 VectorType const & src,
                                                 Number const       evaluation_time) const
{
  time = evaluation_time;

  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop,
                    &This::boundary_face_loop_full,
                    this,
                    dst,
                    src,
                    true /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::evaluate_add(VectorType &       dst,
                                                     VectorType const & src,
                                                     Number const       evaluation_time) const
{
  time = evaluation_time;

  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop,
                    &This::boundary_face_loop_full,
                    this,
                    dst,
                    src,
                    false /*zero_dst_vector = false*/);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::cell_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::face_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  IntegratorFace integrator_m(matrix_free, true, data.dof_index, data.quad_index);
  IntegratorFace integrator_p(matrix_free, false, data.dof_index, data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.gather_evaluate(src, true, false);
    integrator_p.gather_evaluate(src, true, false);

    kernel->reinit_face(integrator_m, integrator_p);

    do_face_integral(integrator_m, integrator_p);

    integrator_m.integrate_scatter(true, false, dst);
    integrator_p.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::face_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)face_range;
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::boundary_face_loop_hom(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if(data.use_boundary_data)
  {
    IntegratorFace integrator_m(matrix_free, true, data.dof_index, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);

      integrator_m.gather_evaluate(src, true, false);

      kernel->reinit_boundary_face(integrator_m);

      do_boundary_integral(integrator_m,
                           OperatorType::homogeneous,
                           matrix_free.get_boundary_id(face));

      integrator_m.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::boundary_face_loop_full(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if(data.use_boundary_data)
  {
    IntegratorFace integrator_m(matrix_free, true, data.dof_index, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);

      integrator_m.gather_evaluate(src, true, false);

      kernel->reinit_boundary_face(integrator_m);

      do_boundary_integral(integrator_m, OperatorType::full, matrix_free.get_boundary_id(face));

      integrator_m.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::boundary_face_loop_inhom(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  (void)src;

  if(data.use_boundary_data)
  {
    IntegratorFace integrator_m(matrix_free, true, data.dof_index, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);

      kernel->reinit_boundary_face(integrator_m);

      do_boundary_integral(integrator_m,
                           OperatorType::inhomogeneous,
                           matrix_free.get_boundary_id(face));

      integrator_m.integrate_scatter(true, false, dst);
    }
  }
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                         IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux = kernel->calculate_flux(u_m, u_p, normal_m);

    integrator_m.submit_value(flux, q);
    integrator_p.submit_value(-flux, q);
  }
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = calculate_interior_value(q, integrator_m, operator_type);
    vector u_p = calculate_exterior_value(
      u_m, q, integrator_m, operator_type, boundary_type, boundary_id, data.bc, time);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux = kernel->calculate_flux(u_m, u_p, normal_m);

    integrator_m.submit_value(flux, q);
  }
}

template class ContinuityPenaltyOperator<2, float>;
template class ContinuityPenaltyOperator<2, double>;

template class ContinuityPenaltyOperator<3, float>;
template class ContinuityPenaltyOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
