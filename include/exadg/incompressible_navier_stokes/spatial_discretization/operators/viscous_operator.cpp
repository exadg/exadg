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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
void
ViscousOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free,
  dealii::AffineConstraints<Number> const &              affine_constraints,
  ViscousOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel)
{
  operator_data = data;

  kernel = viscous_kernel;

  Base::reinit(matrix_free, affine_constraints, data);

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
ViscousOperator<dim, Number>::reinit_face_derived(IntegratorFace &   integrator_m,
                                                  IntegratorFace &   integrator_p,
                                                  unsigned int const face) const
{
  (void)face;

  kernel->reinit_face(integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::reinit_boundary_face_derived(IntegratorFace &   integrator_m,
                                                           unsigned int const face) const
{
  (void)face;

  kernel->reinit_boundary_face(integrator_m, operator_data.dof_index);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::reinit_face_cell_based_derived(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  (void)cell;
  (void)face;

  kernel->reinit_face_cell_based(boundary_id, integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    scalar viscosity = kernel->get_viscosity_cell(integrator.get_current_cell_index(), q);
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
      kernel->get_viscosity_interior_face(integrator_m.get_current_cell_index(), q);
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
      kernel->get_viscosity_interior_face(integrator_m.get_current_cell_index(), q);
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
    // multiply by -1.0 to get the correct normal vector !
    vector normal_p = -integrator_p.get_normal_vector(q);

    scalar average_viscosity =
      kernel->get_viscosity_interior_face(integrator_p.get_current_cell_index(), q);
    tensor gradient_flux =
      kernel->calculate_gradient_flux(value_p, value_m, normal_p, average_viscosity);

    // set exterior gradient to zero
    vector normal_gradient_m;
    // multiply by -1.0 since normal vector n⁺ = -n⁻ !
    vector normal_gradient_p = -kernel->calculate_normal_gradient(q, integrator_p);

    vector value_flux = kernel->calculate_value_flux(
      normal_gradient_p, normal_gradient_m, value_p, value_m, normal_p, average_viscosity);

    integrator_p.submit_gradient(gradient_flux, q);
    integrator_p.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number>
void
ViscousOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
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
                                              this->time);

    vector normal = integrator.get_normal_vector(q);

    scalar viscosity = kernel->get_viscosity_boundary_face(integrator.get_current_cell_index(), q);
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
} // namespace ExaDG
