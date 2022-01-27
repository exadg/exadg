/*
 * projection_operator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/projection_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &        matrix_free,
  dealii::AffineConstraints<Number> const &      affine_constraints,
  ProjectionOperatorData<dim> const &            data,
  Operators::DivergencePenaltyKernelData const & div_kernel_data,
  Operators::ContinuityPenaltyKernelData const & conti_kernel_data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  if(operator_data.use_divergence_penalty)
  {
    this->div_kernel = std::make_shared<Operators::DivergencePenaltyKernel<dim, Number>>();
    this->div_kernel->reinit(matrix_free,
                             operator_data.dof_index,
                             operator_data.quad_index,
                             div_kernel_data);
  }

  if(operator_data.use_continuity_penalty)
  {
    this->conti_kernel = std::make_shared<Operators::ContinuityPenaltyKernel<dim, Number>>();
    this->conti_kernel->reinit(matrix_free,
                               operator_data.dof_index,
                               operator_data.quad_index,
                               conti_kernel_data);
  }

  // mass operator
  this->integrator_flags.cell_evaluate  = CellFlags(true, false, false);
  this->integrator_flags.cell_integrate = CellFlags(true, false, false);

  // divergence penalty
  if(operator_data.use_divergence_penalty)
    this->integrator_flags = this->integrator_flags || div_kernel->get_integrator_flags();

  // continuity penalty
  if(operator_data.use_continuity_penalty)
    this->integrator_flags = this->integrator_flags || conti_kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  ProjectionOperatorData<dim> const &       data,
  std::shared_ptr<DivKernel>                div_penalty_kernel,
  std::shared_ptr<ContiKernel>              conti_penalty_kernel)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  div_kernel   = div_penalty_kernel;
  conti_kernel = conti_penalty_kernel;

  // mass operator
  this->integrator_flags.cell_evaluate  = CellFlags(true, false, false);
  this->integrator_flags.cell_integrate = CellFlags(true, false, false);

  // divergence penalty
  if(operator_data.use_divergence_penalty)
    this->integrator_flags = this->integrator_flags || div_kernel->get_integrator_flags();

  // continuity penalty
  if(operator_data.use_continuity_penalty)
    this->integrator_flags = this->integrator_flags || conti_kernel->get_integrator_flags();
}

template<int dim, typename Number>
ProjectionOperatorData<dim>
ProjectionOperator<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
Operators::DivergencePenaltyKernelData
ProjectionOperator<dim, Number>::get_divergence_kernel_data() const
{
  if(operator_data.use_divergence_penalty)
    return div_kernel->get_data();
  else
    return Operators::DivergencePenaltyKernelData();
}

template<int dim, typename Number>
Operators::ContinuityPenaltyKernelData
ProjectionOperator<dim, Number>::get_continuity_kernel_data() const
{
  if(operator_data.use_continuity_penalty)
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
dealii::LinearAlgebra::distributed::Vector<Number> const &
ProjectionOperator<dim, Number>::get_velocity() const
{
  AssertThrow(velocity != nullptr,
              dealii::ExcMessage("Velocity ptr is not initialized in ProjectionOperator."));

  return *velocity;
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::update(VectorType const & velocity, double const & dt)
{
  this->velocity = &velocity;

  if(operator_data.use_divergence_penalty)
    div_kernel->calculate_penalty_parameter(velocity);
  if(operator_data.use_continuity_penalty)
    conti_kernel->calculate_penalty_parameter(velocity);

  time_step_size = dt;
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(operator_data.use_divergence_penalty)
    div_kernel->reinit_cell(*this->integrator);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(operator_data.use_continuity_penalty)
    conti_kernel->reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  conti_kernel->reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_face_cell_based(
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(operator_data.use_continuity_penalty)
    conti_kernel->reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(integrator.get_value(q), q);

    if(operator_data.use_divergence_penalty)
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
ProjectionOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  if(operator_data.use_boundary_data == true)
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m      = calculate_interior_value(q, integrator_m, operator_type);
      vector u_p      = calculate_exterior_value(u_m,
                                            q,
                                            integrator_m,
                                            operator_type,
                                            boundary_type,
                                            boundary_id,
                                            operator_data.bc,
                                            this->time);
      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

      integrator_m.submit_value(flux, q);
    }
  }
  else
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector flux; // continuity penalty term is zero on boundary faces if u_p = u_m

      integrator_m.submit_value(flux, q);
    }
  }
}

template class ProjectionOperator<2, float>;
template class ProjectionOperator<2, double>;

template class ProjectionOperator<3, float>;
template class ProjectionOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
