/*
 * projection_operator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "projection_operator.h"

namespace IncNS
{
template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ProjectionOperatorData<dim> const & data)
{
  (void)matrix_free;
  (void)constraint_matrix;
  (void)data;

  AssertThrow(false,
              ExcMessage("This reinit function is not implemented for projection operator."));
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &                matrix_free,
  AffineConstraints<double> const &              constraint_matrix,
  ProjectionOperatorData<dim> const &            data,
  Operators::DivergencePenaltyKernelData const & div_kernel_data,
  Operators::ContinuityPenaltyKernelData const & conti_kernel_data)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  if(this->data.use_divergence_penalty)
  {
    this->div_kernel.reset(new Operators::DivergencePenaltyKernel<dim, Number>());
    this->div_kernel->reinit(matrix_free,
                             this->data.dof_index,
                             this->data.quad_index,
                             div_kernel_data);
  }

  if(this->data.use_continuity_penalty)
  {
    this->conti_kernel.reset(new Operators::ContinuityPenaltyKernel<dim, Number>());
    this->conti_kernel->reinit(matrix_free,
                               this->data.dof_index,
                               this->data.quad_index,
                               conti_kernel_data);
  }

  // mass matrix
  this->integrator_flags.cell_evaluate  = CellFlags(true, false, false);
  this->integrator_flags.cell_integrate = CellFlags(true, false, false);

  // divergence penalty
  if(this->data.use_divergence_penalty)
    this->integrator_flags = this->integrator_flags || div_kernel->get_integrator_flags();

  // continuity penalty
  if(this->data.use_continuity_penalty)
    this->integrator_flags = this->integrator_flags || conti_kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ProjectionOperatorData<dim> const & data,
                                        std::shared_ptr<DivKernel>          div_penalty_kernel,
                                        std::shared_ptr<ContiKernel>        conti_penalty_kernel)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  div_kernel   = div_penalty_kernel;
  conti_kernel = conti_penalty_kernel;

  // mass matrix
  this->integrator_flags.cell_evaluate  = CellFlags(true, false, false);
  this->integrator_flags.cell_integrate = CellFlags(true, false, false);

  // divergence penalty
  if(this->data.use_divergence_penalty)
    this->integrator_flags = this->integrator_flags || div_kernel->get_integrator_flags();

  // continuity penalty
  if(this->data.use_continuity_penalty)
    this->integrator_flags = this->integrator_flags || conti_kernel->get_integrator_flags();
}

template<int dim, typename Number>
ProjectionOperatorData<dim>
ProjectionOperator<dim, Number>::get_data() const
{
  return this->data;
}

template<int dim, typename Number>
Operators::DivergencePenaltyKernelData
ProjectionOperator<dim, Number>::get_divergence_kernel_data() const
{
  if(this->data.use_divergence_penalty)
    return div_kernel->get_data();
  else
    return Operators::DivergencePenaltyKernelData();
}

template<int dim, typename Number>
Operators::ContinuityPenaltyKernelData
ProjectionOperator<dim, Number>::get_continuity_kernel_data() const
{
  if(this->data.use_continuity_penalty)
    return conti_kernel->get_data();
  else
    return Operators::ContinuityPenaltyKernelData();
}

template<int dim, typename Number>
double
ProjectionOperator<dim, Number>::get_time_step_size() const
{
  return this->time_step_size;
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
ProjectionOperator<dim, Number>::get_velocity() const
{
  AssertThrow(velocity != nullptr,
              ExcMessage("Velocity ptr is not initialized in ProjectionOperator."));

  return *velocity;
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::update(VectorType const & velocity, double const & dt)
{
  this->velocity = &velocity;

  if(this->data.use_divergence_penalty)
    div_kernel->calculate_penalty_parameter(velocity);
  if(this->data.use_continuity_penalty)
    conti_kernel->calculate_penalty_parameter(velocity);

  time_step_size = dt;
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(this->data.use_divergence_penalty)
    div_kernel->reinit_cell(*this->integrator);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(this->data.use_continuity_penalty)
    conti_kernel->reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                        unsigned int const       face,
                                                        types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(this->data.use_continuity_penalty)
    conti_kernel->reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(integrator.get_value(q), q);

    if(this->data.use_divergence_penalty)
      integrator.submit_divergence(time_step_size * div_kernel->get_volume_flux(integrator, q), q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

    integrator_m.submit_value(flux, q);
    integrator_p.submit_value(-flux, q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = integrator_m.get_value(q);
    vector u_p; // set u_p to zero
    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    vector u_m; // set u_m to zero
    vector u_p      = integrator_p.get_value(q);
    vector normal_p = -integrator_p.get_normal_vector(q);

    vector flux = time_step_size * conti_kernel->calculate_flux(u_p, u_m, normal_p);

    integrator_p.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator_m,
                                                      OperatorType const &       operator_type,
                                                      types::boundary_id const & boundary_id) const
{
  (void)operator_type;
  (void)boundary_id;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector flux; // continuity penalty term is zero on boundary faces

    integrator_m.submit_value(flux, q);
  }
}

template class ProjectionOperator<2, float>;
template class ProjectionOperator<2, double>;

template class ProjectionOperator<3, float>;
template class ProjectionOperator<3, double>;

} // namespace IncNS
