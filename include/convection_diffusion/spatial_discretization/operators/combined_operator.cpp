/*
 * convection_diffusion_operator_merged.cpp
 *
 *  Created on: Jun 6, 2019
 *      Author: fehn
 */

#include "combined_operator.h"
#include "verify_boundary_conditions.h"
#include "weak_boundary_conditions.h"

namespace ConvDiff
{
template<int dim, typename Number>
Operator<dim, Number>::Operator() : scaling_factor_mass_matrix(1.0)
{
}

template<int dim, typename Number>
void
Operator<dim, Number>::reinit(MatrixFree<dim, Number> const &   matrix_free,
                              AffineConstraints<double> const & constraint_matrix,
                              OperatorData<dim> const &         data)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  if(this->data.convective_problem)
    convective_kernel.reinit(matrix_free,
                             data.convective_kernel_data,
                             data.quad_index,
                             this->is_mg);

  if(this->data.diffusive_problem)
    diffusive_kernel.reinit(matrix_free, data.diffusive_kernel_data, data.dof_index);

  if(this->data.unsteady_problem)
    this->integrator_flags = this->integrator_flags || mass_kernel.get_integrator_flags();
  if(this->data.convective_problem)
    this->integrator_flags = this->integrator_flags || convective_kernel.get_integrator_flags();
  if(this->data.diffusive_problem)
    this->integrator_flags = this->integrator_flags || diffusive_kernel.get_integrator_flags();
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_after_mesh_movement()
{
  diffusive_kernel.calculate_penalty_parameter(*this->matrix_free, this->data.dof_index);
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
Operator<dim, Number>::get_velocity() const
{
  return convective_kernel.get_velocity();
}

template<int dim, typename Number>
void
Operator<dim, Number>::set_velocity_copy(VectorType const & velocity) const
{
  if(this->data.convective_problem)
    convective_kernel.set_velocity_copy(velocity);
}

template<int dim, typename Number>
void
Operator<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  if(this->data.convective_problem)
    convective_kernel.set_velocity_ptr(velocity);
}

template<int dim, typename Number>
Number
Operator<dim, Number>::get_scaling_factor_mass_matrix() const
{
  return scaling_factor_mass_matrix;
}

template<int dim, typename Number>
void
Operator<dim, Number>::set_scaling_factor_mass_matrix(Number const & number)
{
  scaling_factor_mass_matrix = number;
}

template<int dim, typename Number>
void
Operator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(this->data.convective_problem)
    convective_kernel.reinit_cell(cell);
}

template<int dim, typename Number>
void
Operator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(this->data.convective_problem)
    convective_kernel.reinit_face(face);
  if(this->data.diffusive_problem)
    diffusive_kernel.reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
Operator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  if(this->data.convective_problem)
    convective_kernel.reinit_boundary_face(face);
  if(this->data.diffusive_problem)
    diffusive_kernel.reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
Operator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                              unsigned int const       face,
                                              types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(this->data.convective_problem)
    convective_kernel.reinit_face_cell_based(cell, face, boundary_id);
  if(this->data.diffusive_problem)
    diffusive_kernel.reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  bool const get_value =
    this->data.unsteady_problem ||
    (this->data.convective_problem && this->data.convective_kernel_data.formulation ==
                                        FormulationConvectiveTerm::DivergenceFormulation);

  bool const get_gradient =
    this->data.diffusive_problem ||
    (this->data.convective_problem && this->data.convective_kernel_data.formulation ==
                                        FormulationConvectiveTerm::ConvectiveFormulation);

  bool const submit_value =
    this->data.unsteady_problem ||
    (this->data.convective_problem && this->data.convective_kernel_data.formulation ==
                                        FormulationConvectiveTerm::ConvectiveFormulation);

  bool const submit_gradient =
    this->data.diffusive_problem ||
    (this->data.convective_problem && this->data.convective_kernel_data.formulation ==
                                        FormulationConvectiveTerm::DivergenceFormulation);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector gradient_flux;
    scalar value_flux = make_vectorized_array<Number>(0.0);

    scalar value = make_vectorized_array<Number>(0.0);
    if(get_value)
      value = integrator.get_value(q);

    vector gradient;
    if(get_gradient)
      gradient = integrator.get_gradient(q);

    if(this->data.unsteady_problem)
    {
      value_flux += mass_kernel.get_volume_flux(scaling_factor_mass_matrix, value);
    }

    if(this->data.convective_problem)
    {
      if(this->data.convective_kernel_data.formulation ==
         FormulationConvectiveTerm::DivergenceFormulation)
      {
        gradient_flux +=
          convective_kernel.get_volume_flux_divergence_form(value, integrator, q, this->time);
      }
      else if(this->data.convective_kernel_data.formulation ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        value_flux +=
          convective_kernel.get_volume_flux_convective_form(gradient, integrator, q, this->time);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }
    }

    if(this->data.diffusive_problem)
    {
      gradient_flux += diffusive_kernel.get_volume_flux(integrator, q);
    }

    if(submit_value)
      integrator.submit_value(value_flux, q);
    if(submit_gradient)
      integrator.submit_gradient(gradient_flux, q);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                        IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux_m = make_vectorized_array<Number>(0.0);
    scalar value_flux_p = make_vectorized_array<Number>(0.0);

    if(this->data.convective_problem)
    {
      std::tuple<scalar, scalar> flux = convective_kernel.calculate_flux_interior_and_neighbor(
        q, integrator_m, value_m, value_p, normal_m, this->time, true);

      value_flux_m += std::get<0>(flux);
      value_flux_p += std::get<1>(flux);
    }

    if(this->data.diffusive_problem)
    {
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = integrator_p.get_normal_derivative(q);

      scalar value_flux = diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                                normal_gradient_p,
                                                                value_m,
                                                                value_p);

      value_flux_m += -value_flux;
      value_flux_p += value_flux; // + sign since n⁺ = -n⁻
    }

    integrator_m.submit_value(value_flux_m, q);
    integrator_p.submit_value(value_flux_p, q);

    if(this->data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
      integrator_p.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                            IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->data.convective_problem)
    {
      value_flux += convective_kernel.calculate_flux_interior(
        q, integrator_m, value_m, value_p, normal_m, this->time, true);
    }

    if(this->data.diffusive_problem)
    {
      // set exterior value to zero
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(this->data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

// TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
// cell-based face loops
template<int dim, typename Number>
void
Operator<dim, Number>::do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                                       IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->data.convective_problem)
    {
      // TODO
      // The matrix-free implementation in deal.II does currently not allow to access neighboring
      // data in case of cell-based face loops. We therefore have to use integrator_velocity_m twice
      // to avoid the problem of accessing data of the neighboring element. Note that this variant
      // calculates the diagonal and block-diagonal only approximately. The theoretically correct
      // version using integrator_velocity_p is currently not implemented in deal.II.
      bool exterior_velocity_available =
        false; // TODO -> set to true once functionality is available
      value_flux += convective_kernel.calculate_flux_interior(
        q, integrator_m, value_m, value_p, normal_m, this->time, exterior_velocity_available);
    }

    if(this->data.diffusive_problem)
    {
      // set exterior value to zero
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(this->data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                            IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m = make_vectorized_array<Number>(0.0);
    scalar value_p = integrator_p.get_value(q);

    // n⁺ = -n⁻
    vector normal_p = -integrator_p.get_normal_vector(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->data.convective_problem)
    {
      // minus sign for convective flux since n⁺ = -n⁻
      value_flux += convective_kernel.calculate_flux_interior(
        q, integrator_p, value_p, value_m, normal_p, this->time, true);
    }

    if(this->data.diffusive_problem)
    {
      // set gradient_m to zero
      scalar normal_gradient_m = make_vectorized_array<Number>(0.0);
      // minus sign to get the correct normal vector n⁺ = -n⁻
      scalar normal_gradient_p = -integrator_p.get_normal_derivative(q);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_p,
                                                           normal_gradient_m,
                                                           value_p,
                                                           value_m);
    }

    integrator_p.submit_value(value_flux, q);

    if(this->data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_p, value_m);
      // opposite sign since n⁺ = -n⁻
      integrator_p.submit_normal_derivative(-gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator_m,
                                            OperatorType const &       operator_type,
                                            types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, integrator_m, operator_type);
    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              integrator_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              this->data.bc,
                                              this->time);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->data.convective_problem)
    {
      // In case of numerical velocity field:
      // Simply use velocity_p = velocity_m on boundary faces
      // -> exterior_velocity_available = false.
      value_flux += convective_kernel.calculate_flux_interior(
        q, integrator_m, value_m, value_p, normal_m, this->time, false);
    }

    if(this->data.diffusive_problem)
    {
      scalar normal_gradient_m = calculate_interior_normal_gradient(q, integrator_m, operator_type);
      scalar normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                    q,
                                                                    integrator_m,
                                                                    operator_type,
                                                                    boundary_type,
                                                                    boundary_id,
                                                                    this->data.bc,
                                                                    this->time);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(this->data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  OperatorData<dim> const &            data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  do_verify_boundary_conditions(boundary_id, data, periodic_boundary_ids);
}

template class Operator<2, float>;
template class Operator<2, double>;

template class Operator<3, float>;
template class Operator<3, double>;

} // namespace ConvDiff
