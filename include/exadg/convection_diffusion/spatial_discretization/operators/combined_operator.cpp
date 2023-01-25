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

#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
CombinedOperator<dim, Number>::CombinedOperator() : scaling_factor_mass(1.0)
{
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  CombinedOperatorData<dim> const &         data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  if(operator_data.unsteady_problem)
  {
    mass_kernel = std::make_shared<MassKernel<dim, Number>>();
  }

  if(operator_data.convective_problem)
  {
    convective_kernel = std::make_shared<Operators::ConvectiveKernel<dim, Number>>();
    convective_kernel->reinit(matrix_free,
                              data.convective_kernel_data,
                              data.quad_index,
                              this->is_mg);
  }

  if(operator_data.diffusive_problem)
  {
    diffusive_kernel = std::make_shared<Operators::DiffusiveKernel<dim, Number>>();
    diffusive_kernel->reinit(matrix_free, data.diffusive_kernel_data, data.dof_index);
  }

  // integrator flags
  if(operator_data.unsteady_problem)
    this->integrator_flags = this->integrator_flags | mass_kernel->get_integrator_flags();

  if(operator_data.convective_problem)
    this->integrator_flags = this->integrator_flags | convective_kernel->get_integrator_flags();

  if(operator_data.diffusive_problem)
    this->integrator_flags = this->integrator_flags | diffusive_kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                   matrix_free,
  dealii::AffineConstraints<Number> const &                 affine_constraints,
  CombinedOperatorData<dim> const &                         data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel_in,
  std::shared_ptr<Operators::DiffusiveKernel<dim, Number>>  diffusive_kernel_in)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  if(operator_data.unsteady_problem)
    mass_kernel = std::make_shared<MassKernel<dim, Number>>();

  if(operator_data.convective_problem)
    convective_kernel = convective_kernel_in;

  if(operator_data.diffusive_problem)
    diffusive_kernel = diffusive_kernel_in;

  // integrator flags
  if(operator_data.unsteady_problem)
    this->integrator_flags = this->integrator_flags | mass_kernel->get_integrator_flags();

  if(operator_data.convective_problem)
    this->integrator_flags = this->integrator_flags | convective_kernel->get_integrator_flags();

  if(operator_data.diffusive_problem)
    this->integrator_flags = this->integrator_flags | diffusive_kernel->get_integrator_flags();
}

template<int dim, typename Number>
CombinedOperatorData<dim> const &
CombinedOperator<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::update_after_grid_motion()
{
  if(operator_data.diffusive_problem)
    diffusive_kernel->calculate_penalty_parameter(*this->matrix_free, operator_data.dof_index);
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
CombinedOperator<dim, Number>::get_velocity() const
{
  return convective_kernel->get_velocity();
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::set_velocity_copy(VectorType const & velocity) const
{
  if(operator_data.convective_problem)
    convective_kernel->set_velocity_copy(velocity);
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  if(operator_data.convective_problem)
    convective_kernel->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
Number
CombinedOperator<dim, Number>::get_scaling_factor_mass_operator() const
{
  return scaling_factor_mass;
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::set_scaling_factor_mass_operator(Number const & scaling_factor)
{
  scaling_factor_mass = scaling_factor;
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(operator_data.convective_problem)
    convective_kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(operator_data.convective_problem)
    convective_kernel->reinit_face(face);
  if(operator_data.diffusive_problem)
    diffusive_kernel->reinit_face(*this->integrator_m,
                                  *this->integrator_p,
                                  operator_data.dof_index);
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  if(operator_data.convective_problem)
    convective_kernel->reinit_boundary_face(face);
  if(operator_data.diffusive_problem)
    diffusive_kernel->reinit_boundary_face(*this->integrator_m, operator_data.dof_index);
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::reinit_face_cell_based(
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(operator_data.convective_problem)
    convective_kernel->reinit_face_cell_based(cell, face, boundary_id);
  if(operator_data.diffusive_problem)
    diffusive_kernel->reinit_face_cell_based(boundary_id,
                                             *this->integrator_m,
                                             *this->integrator_p,
                                             operator_data.dof_index);
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector gradient_flux;
    scalar value_flux = dealii::make_vectorized_array<Number>(0.0);

    scalar value = dealii::make_vectorized_array<Number>(0.0);
    if(this->integrator_flags.cell_evaluate & dealii::EvaluationFlags::values)
      value = integrator.get_value(q);

    vector gradient;
    if(this->integrator_flags.cell_evaluate & dealii::EvaluationFlags::gradients)
      gradient = integrator.get_gradient(q);

    if(operator_data.unsteady_problem)
    {
      value_flux += mass_kernel->get_volume_flux(scaling_factor_mass, value);
    }

    if(operator_data.convective_problem)
    {
      if(operator_data.convective_kernel_data.formulation ==
         FormulationConvectiveTerm::DivergenceFormulation)
      {
        gradient_flux +=
          convective_kernel->get_volume_flux_divergence_form(value, integrator, q, this->time);
      }
      else if(operator_data.convective_kernel_data.formulation ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        value_flux +=
          convective_kernel->get_volume_flux_convective_form(gradient, integrator, q, this->time);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }

    if(operator_data.diffusive_problem)
    {
      gradient_flux += diffusive_kernel->get_volume_flux(integrator, q);
    }

    if(this->integrator_flags.cell_integrate & dealii::EvaluationFlags::values)
      integrator.submit_value(value_flux, q);
    if(this->integrator_flags.cell_integrate & dealii::EvaluationFlags::gradients)
      integrator.submit_gradient(gradient_flux, q);
  }
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux_m = dealii::make_vectorized_array<Number>(0.0);
    scalar value_flux_p = dealii::make_vectorized_array<Number>(0.0);

    if(operator_data.convective_problem)
    {
      std::tuple<scalar, scalar> flux = convective_kernel->calculate_flux_interior_and_neighbor(
        q, integrator_m, value_m, value_p, normal_m, this->time, true);

      value_flux_m += std::get<0>(flux);
      value_flux_p += std::get<1>(flux);
    }

    if(operator_data.diffusive_problem)
    {
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = integrator_p.get_normal_derivative(q);

      scalar value_flux = diffusive_kernel->calculate_value_flux(normal_gradient_m,
                                                                 normal_gradient_p,
                                                                 value_m,
                                                                 value_p);

      value_flux_m += -value_flux;
      value_flux_p += value_flux; // + sign since n⁺ = -n⁻
    }

    integrator_m.submit_value(value_flux_m, q);
    integrator_p.submit_value(value_flux_p, q);

    if(operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel->calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
      integrator_p.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                    IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = dealii::make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux = dealii::make_vectorized_array<Number>(0.0);

    if(operator_data.convective_problem)
    {
      value_flux += convective_kernel->calculate_flux_interior(
        q, integrator_m, value_m, value_p, normal_m, this->time, true);
    }

    if(operator_data.diffusive_problem)
    {
      // set exterior value to zero
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = dealii::make_vectorized_array<Number>(0.0);

      value_flux += -diffusive_kernel->calculate_value_flux(normal_gradient_m,
                                                            normal_gradient_p,
                                                            value_m,
                                                            value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel->calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

// TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
// cell-based face loops
template<int dim, typename Number>
void
CombinedOperator<dim, Number>::do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                                               IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = dealii::make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux = dealii::make_vectorized_array<Number>(0.0);

    if(operator_data.convective_problem)
    {
      // TODO
      // The matrix-free implementation in deal.II does currently not allow to access neighboring
      // data in case of cell-based face loops. We therefore have to use integrator_velocity_m twice
      // to avoid the problem of accessing data of the neighboring element. Note that this variant
      // calculates the diagonal and block-diagonal only approximately. The theoretically correct
      // version using integrator_velocity_p is currently not implemented in deal.II.
      bool exterior_velocity_available =
        false; // TODO -> set to true once functionality is available
      value_flux += convective_kernel->calculate_flux_interior(
        q, integrator_m, value_m, value_p, normal_m, this->time, exterior_velocity_available);
    }

    if(operator_data.diffusive_problem)
    {
      // set exterior value to zero
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = dealii::make_vectorized_array<Number>(0.0);

      value_flux += -diffusive_kernel->calculate_value_flux(normal_gradient_m,
                                                            normal_gradient_p,
                                                            value_m,
                                                            value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel->calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                    IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m = dealii::make_vectorized_array<Number>(0.0);
    scalar value_p = integrator_p.get_value(q);

    // n⁺ = -n⁻
    vector normal_p = -integrator_p.get_normal_vector(q);

    scalar value_flux = dealii::make_vectorized_array<Number>(0.0);

    if(operator_data.convective_problem)
    {
      value_flux += convective_kernel->calculate_flux_interior(
        q, integrator_p, value_p, value_m, normal_p, this->time, true);
    }

    if(operator_data.diffusive_problem)
    {
      // set gradient_m to zero
      scalar normal_gradient_m = dealii::make_vectorized_array<Number>(0.0);
      // minus sign to get the correct normal vector n⁺ = -n⁻
      scalar normal_gradient_p = -integrator_p.get_normal_derivative(q);

      value_flux += -diffusive_kernel->calculate_value_flux(normal_gradient_p,
                                                            normal_gradient_m,
                                                            value_p,
                                                            value_m);
    }

    integrator_p.submit_value(value_flux, q);

    if(operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel->calculate_gradient_flux(value_p, value_m);
      // opposite sign since n⁺ = -n⁻
      integrator_p.submit_normal_derivative(-gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
CombinedOperator<dim, Number>::do_boundary_integral(
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

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux = dealii::make_vectorized_array<Number>(0.0);

    if(operator_data.convective_problem)
    {
      // In case of numerical velocity field:
      // Simply use velocity_p = velocity_m on boundary faces
      // -> exterior_velocity_available = false.
      value_flux += convective_kernel->calculate_flux_interior(
        q, integrator_m, value_m, value_p, normal_m, this->time, false);
    }

    if(operator_data.diffusive_problem)
    {
      scalar normal_gradient_m = calculate_interior_normal_gradient(q, integrator_m, operator_type);
      scalar normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                    q,
                                                                    integrator_m,
                                                                    operator_type,
                                                                    boundary_type,
                                                                    boundary_id,
                                                                    operator_data.bc,
                                                                    this->time);

      value_flux += -diffusive_kernel->calculate_value_flux(normal_gradient_m,
                                                            normal_gradient_p,
                                                            value_m,
                                                            value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel->calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template class CombinedOperator<2, float>;
template class CombinedOperator<2, double>;

template class CombinedOperator<3, float>;
template class CombinedOperator<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
