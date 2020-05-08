/*
 * viscous_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */


#include "viscous_operator.h"

namespace IncNS
{
template<int dim, typename Number>
void
ViscousOperator<dim, Number>::initialize(
  MatrixFree<dim, Number> const &                        matrix_free,
  AffineConstraints<double> const &                      constraint_matrix,
  ViscousOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel)
{
  operator_data = data;

  kernel = viscous_kernel;

  Base::reinit(matrix_free, constraint_matrix, data);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::update()
{
  kernel->calculate_penalty_parameter(this->get_matrix_free(), operator_data.dof_index);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel->reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel->reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                     unsigned int const       face,
                                                     types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel->reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    scalar viscosity = kernel->get_viscosity_cell(integrator.get_cell_index(), q);
    integrator.submit_gradient(kernel->get_volume_flux(integrator.get_gradient(q), viscosity), q);
  }
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                               IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector value_m = integrator_m.get_value(q);
    vector value_p = integrator_p.get_value(q);
    vector normal  = integrator_m.get_normal_vector(q);

    scalar average_viscosity =
      kernel->get_viscosity_interior_face(integrator_m.get_face_index(), q);
    tensor gradient_flux =
      kernel->calculate_gradient_flux(value_m, value_p, normal, average_viscosity);

    vector normal_gradient_m = kernel->calculate_normal_gradient(q, integrator_m);
    vector normal_gradient_p = kernel->calculate_normal_gradient(q, integrator_p);

    vector value_flux = kernel->calculate_value_flux(
      normal_gradient_m, normal_gradient_p, value_m, value_p, normal, average_viscosity);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_p.submit_gradient(gradient_flux, q);

    integrator_m.submit_value(-value_flux, q);
    integrator_p.submit_value(value_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                   IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector value_m = integrator_m.get_value(q);
    vector value_p; // set exterior values to zero
    vector normal_m = integrator_m.get_normal_vector(q);

    scalar average_viscosity =
      kernel->get_viscosity_interior_face(integrator_m.get_face_index(), q);
    tensor gradient_flux =
      kernel->calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

    vector normal_gradient_m = kernel->calculate_normal_gradient(q, integrator_m);
    vector normal_gradient_p; // set exterior gradient to zero

    vector value_flux = kernel->calculate_value_flux(
      normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                   IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    vector value_m; // set exterior values to zero
    vector value_p = integrator_p.get_value(q);
    // multiply by -1.0 to get the correct normal vector !!!
    vector normal_p = -integrator_p.get_normal_vector(q);

    scalar average_viscosity =
      kernel->get_viscosity_interior_face(integrator_p.get_face_index(), q);
    tensor gradient_flux =
      kernel->calculate_gradient_flux(value_p, value_m, normal_p, average_viscosity);

    // set exterior gradient to zero
    vector normal_gradient_m;
    // multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
    vector normal_gradient_p = -kernel->calculate_normal_gradient(q, integrator_p);

    vector value_flux = kernel->calculate_value_flux(
      normal_gradient_p, normal_gradient_m, value_p, value_m, normal_p, average_viscosity);

    integrator_p.submit_gradient(gradient_flux, q);
    integrator_p.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator,
                                                   OperatorType const &       operator_type,
                                                   types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector value_m = calculate_interior_value(q, integrator, operator_type);
    vector value_p = calculate_exterior_value(value_m,
                                              q,
                                              integrator,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              operator_data.bc,
                                              this->time,
                                              operator_data.quad_index);

    vector normal = integrator.get_normal_vector(q);

    scalar viscosity     = kernel->get_viscosity_boundary_face(integrator.get_face_index(), q);
    tensor gradient_flux = kernel->calculate_gradient_flux(value_m, value_p, normal, viscosity);

    vector normal_gradient_m =
      kernel->calculate_interior_normal_gradient(q, integrator, operator_type);

    vector normal_gradient_p;
    normal_gradient_p =
      calculate_exterior_normal_gradient(normal_gradient_m,
                                         q,
                                         integrator,
                                         operator_type,
                                         boundary_type,
                                         boundary_id,
                                         operator_data.bc,
                                         this->time,
                                         kernel->get_data().variable_normal_vector);

    vector value_flux = kernel->calculate_value_flux(
      normal_gradient_m, normal_gradient_p, value_m, value_p, normal, viscosity);

    integrator.submit_gradient(gradient_flux, q);
    integrator.submit_value(-value_flux, q);
  }
}

template class ViscousOperator<2, float>;
template class ViscousOperator<2, double>;

template class ViscousOperator<3, float>;
template class ViscousOperator<3, double>;

} // namespace IncNS
