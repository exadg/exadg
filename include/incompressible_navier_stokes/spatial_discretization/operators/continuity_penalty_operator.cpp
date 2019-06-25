/*
 * continuity_penalty_operator.cpp
 *
 *  Created on: Jun 25, 2019
 *      Author: fehn
 */


#include "continuity_penalty_operator.h"

namespace IncNS
{
template<int dim, typename Number>
ContinuityPenaltyOperator<dim, Number>::ContinuityPenaltyOperator() : matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::reinit(MatrixFree<dim, Number> const & matrix_free,
                                               ContinuityPenaltyData const &   data,
                                               std::shared_ptr<Kernel> const   kernel)
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
                    &This::boundary_face_loop_empty,
                    this,
                    dst,
                    src,
                    true,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop,
                    &This::boundary_face_loop_empty,
                    this,
                    dst,
                    src,
                    false,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::cell_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                                                        VectorType &                    dst,
                                                        VectorType const &              src,
                                                        Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
ContinuityPenaltyOperator<dim, Number>::face_loop(MatrixFree<dim, Number> const & matrix_free,
                                                  VectorType &                    dst,
                                                  VectorType const &              src,
                                                  Range const &                   face_range) const
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
ContinuityPenaltyOperator<dim, Number>::boundary_face_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
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

template class ContinuityPenaltyOperator<2, float>;
template class ContinuityPenaltyOperator<2, double>;

template class ContinuityPenaltyOperator<3, float>;
template class ContinuityPenaltyOperator<3, double>;

} // namespace IncNS
