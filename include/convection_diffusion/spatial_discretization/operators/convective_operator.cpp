#include "convective_operator.h"

#include "weak_boundary_conditions.h"

namespace ConvDiff
{
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ConvectiveOperatorData<dim> const & data)
{
  (void)matrix_free;
  (void)constraint_matrix;
  (void)data;

  AssertThrow(false,
              ExcMessage(
                "This reinit() function can not be used to initialize the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &                           matrix_free,
  AffineConstraints<double> const &                         constraint_matrix,
  ConvectiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> kernel_in)
{
  kernel = kernel_in;

  Base::reinit(matrix_free, constraint_matrix, data);

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
LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return kernel->get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel->reinit_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel->reinit_boundary_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                        unsigned int const       face,
                                                        types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel->reinit_face_cell_based(cell, face, boundary_id);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    if(this->data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      scalar value = integrator.get_value(q);
      integrator.submit_gradient(
        kernel->get_volume_flux_divergence_form(value, integrator, q, this->time), q);
    }
    else if(this->data.kernel_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      vector gradient = integrator.get_gradient(q);
      integrator.submit_value(
        kernel->get_volume_flux_convective_form(gradient, integrator, q, this->time), q);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
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
    scalar value_p = make_vectorized_array<Number>(0.0);
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
    scalar value_p = make_vectorized_array<Number>(0.0);
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
    scalar value_m = make_vectorized_array<Number>(0.0);
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
ConvectiveOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator_m,
                                                      OperatorType const &       operator_type,
                                                      types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value<dim, Number, 1, 0>(q, integrator_m, operator_type);
    scalar value_p = calculate_exterior_value<dim, Number, 1, 0>(value_m,
                                                                 q,
                                                                 integrator_m,
                                                                 operator_type,
                                                                 boundary_type,
                                                                 boundary_id,
                                                                 this->data.bc,
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
