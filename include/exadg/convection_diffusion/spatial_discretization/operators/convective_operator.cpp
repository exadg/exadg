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

#include <exadg/convection_diffusion/spatial_discretization/operators/convective_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                   matrix_free,
  dealii::AffineConstraints<Number> const &                 affine_constraints,
  ConvectiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel)
{
  operator_data = data;

  this->kernel = kernel;

  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_copy(VectorType const & velocity_in) const
{
  kernel->set_velocity_copy(velocity_in);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_ptr(VectorType const & velocity_in) const
{
  kernel->set_velocity_ptr(velocity_in);
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return kernel->get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                     unsigned int const cell) const
{
  (void)integrator;

  kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_derived(IntegratorFace &   integrator_m,
                                                     IntegratorFace &   integrator_p,
                                                     unsigned int const face) const
{
  (void)integrator_m;
  (void)integrator_p;

  kernel->reinit_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face_derived(IntegratorFace &   integrator_m,
                                                              unsigned int const face) const
{
  (void)integrator_m;

  kernel->reinit_boundary_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based_derived(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  (void)integrator_m;
  (void)integrator_p;

  kernel->reinit_face_cell_based(cell, face, boundary_id);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    if(operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      scalar value = integrator.get_value(q);
      integrator.submit_gradient(
        kernel->get_volume_flux_divergence_form(value, integrator, q, this->time), q);
    }
    else if(operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector gradient = integrator.get_gradient(q);
      integrator.submit_value(
        kernel->get_volume_flux_convective_form(gradient, integrator, q, this->time), q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<scalar, scalar> flux = kernel->calculate_flux_interior_and_neighbor(
      q, integrator_m, value_m, value_p, normal_m, this->time, true);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = dealii::make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    scalar flux = kernel->calculate_flux_interior(
      q, integrator_m, value_m, value_p, normal_m, this->time, true);

    integrator_m.submit_value(flux, q);
  }
}

// TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
// cell-based face loops
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral_cell_based(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = dealii::make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    // TODO
    // The matrix-free implementation in deal.II does currently not allow to access neighboring data
    // in case of cell-based face loops. We therefore have to use integrator_velocity_m twice to
    // avoid the problem of accessing data of the neighboring element. Note that this variant
    // calculates the diagonal and block-diagonal only approximately. The theoretically correct
    // version using integrator_velocity_p is currently not implemented in deal.II.
    bool exterior_velocity_available = false; // TODO -> set to true once functionality is available
    scalar flux                      = kernel->calculate_flux_interior(
      q, integrator_m, value_m, value_p, normal_m, this->time, exterior_velocity_available);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
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

    scalar flux = kernel->calculate_flux_interior(
      q, integrator_p, value_p, value_m, normal_p, this->time, true);

    integrator_p.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral(
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

    // In case of numerical velocity field:
    // Simply use velocity_p = velocity_m on boundary faces -> exterior_velocity_available = false.
    scalar flux = kernel->calculate_flux_interior(
      q, integrator_m, value_m, value_p, normal_m, this->time, false);

    integrator_m.submit_value(flux, q);
  }
}

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
