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

#include <exadg/convection_diffusion/spatial_discretization/operators/diffusive_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                  matrix_free,
  dealii::AffineConstraints<Number> const &                affine_constraints,
  DiffusiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::DiffusiveKernel<dim, Number>> kernel)
{
  operator_data = data;

  this->kernel = kernel;

  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::update()
{
  kernel->calculate_penalty_parameter(*this->matrix_free, operator_data.dof_index);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit_face(IntegratorFace &   integrator_m,
                                            IntegratorFace &   integrator_p,
                                            unsigned int const face) const
{
  Base::reinit_face(integrator_m, integrator_p, face);

  kernel->reinit_face(integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit_boundary_face(IntegratorFace &   integrator_m,
                                                     unsigned int const face) const
{
  Base::reinit_boundary_face(integrator_m, face);

  kernel->reinit_boundary_face(integrator_m, operator_data.dof_index);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::reinit_face_cell_based(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(integrator_m, integrator_p, cell, face, boundary_id);

  kernel->reinit_face_cell_based(boundary_id, integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_gradient(kernel->get_volume_flux(integrator, q), q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                 IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = integrator_p.get_value(q);

    scalar gradient_flux = kernel->calculate_gradient_flux(value_m, value_p);

    scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
    scalar normal_gradient_p = integrator_p.get_normal_derivative(q);

    scalar value_flux =
      kernel->calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_p.submit_normal_derivative(gradient_flux, q);

    integrator_m.submit_value(-value_flux, q);
    integrator_p.submit_value(value_flux, q); // opposite sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                     IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set exterior value to zero
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = dealii::make_vectorized_array<Number>(0.0);

    scalar gradient_flux = kernel->calculate_gradient_flux(value_m, value_p);

    // set exterior value to zero
    scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
    scalar normal_gradient_p = dealii::make_vectorized_array<Number>(0.0);

    scalar value_flux =
      kernel->calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                     IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m = dealii::make_vectorized_array<Number>(0.0);
    scalar value_p = integrator_p.get_value(q);

    scalar gradient_flux = kernel->calculate_gradient_flux(value_p, value_m);

    // set gradient_m to zero
    scalar normal_gradient_m = dealii::make_vectorized_array<Number>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    scalar normal_gradient_p = -integrator_p.get_normal_derivative(q);

    scalar value_flux =
      kernel->calculate_value_flux(normal_gradient_p, normal_gradient_m, value_p, value_m);

    integrator_p.submit_normal_derivative(-gradient_flux, q); // opposite sign since n⁺ = -n⁻
    integrator_p.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number>
void
DiffusiveOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, integrator_m, operator_type);

    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              integrator_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              operator_data.bc,
                                              this->time);

    scalar gradient_flux = kernel->calculate_gradient_flux(value_m, value_p);

    scalar normal_gradient_m = calculate_interior_normal_gradient(q, integrator_m, operator_type);

    scalar normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                  q,
                                                                  integrator_m,
                                                                  operator_type,
                                                                  boundary_type,
                                                                  boundary_id,
                                                                  operator_data.bc,
                                                                  this->time);

    scalar value_flux =
      kernel->calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template class DiffusiveOperator<2, float>;
template class DiffusiveOperator<2, double>;

template class DiffusiveOperator<3, float>;
template class DiffusiveOperator<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
