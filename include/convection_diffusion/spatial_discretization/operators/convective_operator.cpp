#include "convective_operator.h"

#include "verify_boundary_conditions.h"
#include "weak_boundary_conditions.h"

namespace ConvDiff
{
namespace Operators
{
template<int dim, typename Number>
void
ConvectiveKernel<dim, Number>::reinit(MatrixFree<dim, Number> const &   matrix_free,
                                      ConvectiveKernelData<dim> const & data_in,
                                      unsigned int const                quad_index) const
{
  data = data_in;

  if(data_in.type_velocity_field == TypeVelocityField::Numerical)
  {
    matrix_free.initialize_dof_vector(velocity, data_in.dof_index_velocity);

    integrator_velocity.reset(
      new CellIntegratorVelocity(matrix_free, data_in.dof_index_velocity, quad_index));

    integrator_velocity_m.reset(
      new FaceIntegratorVelocity(matrix_free, true, data_in.dof_index_velocity, quad_index));

    integrator_velocity_p.reset(
      new FaceIntegratorVelocity(matrix_free, false, data_in.dof_index_velocity, quad_index));
  }
}

template<int dim, typename Number>
IntegratorFlags
ConvectiveKernel<dim, Number>::get_integrator_flags() const
{
  IntegratorFlags flags;

  flags.cell_evaluate  = CellFlags(true, false, false);
  flags.cell_integrate = CellFlags(false, true, false);

  flags.face_evaluate  = FaceFlags(true, false);
  flags.face_integrate = FaceFlags(true, false);

  return flags;
}

template<int dim, typename Number>
MappingFlags
ConvectiveKernel<dim, Number>::get_mapping_flags()
{
  MappingFlags flags;

  flags.cells = update_gradients | update_JxW_values |
                update_quadrature_points; // q-points due to analytical velocity field
  flags.inner_faces = update_JxW_values | update_quadrature_points |
                      update_normal_vectors; // q-points due to analytical velocity field
  flags.boundary_faces = update_JxW_values | update_quadrature_points | update_normal_vectors;

  return flags;
}

template<int dim, typename Number>
void
ConvectiveKernel<dim, Number>::set_velocity(VectorType const & velocity_in) const
{
  AssertThrow(data.type_velocity_field == TypeVelocityField::Numerical,
              ExcMessage("Invalid parameter type_velocity_field."));

  velocity = velocity_in;

  velocity.update_ghost_values();
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> &
ConvectiveKernel<dim, Number>::get_velocity() const
{
  return velocity;
}

template<int dim, typename Number>
void
ConvectiveKernel<dim, Number>::reinit_cell(unsigned int const cell) const
{
  if(data.type_velocity_field == TypeVelocityField::Numerical)
  {
    integrator_velocity->reinit(cell);
    integrator_velocity->gather_evaluate(velocity, true, false, false);
  }
}

template<int dim, typename Number>
void
ConvectiveKernel<dim, Number>::reinit_face(unsigned int const face) const
{
  if(data.type_velocity_field == TypeVelocityField::Numerical)
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);

    integrator_velocity_p->reinit(face);
    integrator_velocity_p->gather_evaluate(velocity, true, false);
  }
}

template<int dim, typename Number>
void
ConvectiveKernel<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  if(data.type_velocity_field == TypeVelocityField::Numerical)
  {
    integrator_velocity_m->reinit(face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);
  }
}

template<int dim, typename Number>
void
ConvectiveKernel<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                      unsigned int const       face,
                                                      types::boundary_id const boundary_id) const
{
  if(data.type_velocity_field == TypeVelocityField::Numerical)
  {
    integrator_velocity_m->reinit(cell, face);
    integrator_velocity_m->gather_evaluate(velocity, true, false);

    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      // TODO: Matrix-free implementation in deal.II does currently not allow to access data of the
      // neighboring element in case of cell-based face loops.
      //      integrator_velocity_p->reinit(cell, face);
      //      integrator_velocity_p->gather_evaluate(velocity, true, false);
    }
  }
}

/*
 *  This function calculates the numerical flux using the central flux.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveKernel<dim, Number>::calculate_central_flux(scalar const & value_m,
                                                        scalar const & value_p,
                                                        scalar const & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);

  return normal_velocity * average_value;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveKernel<dim, Number>::calculate_central_flux(scalar const & value_m,
                                                        scalar const & value_p,
                                                        scalar const & normal_velocity_m,
                                                        scalar const & normal_velocity_p) const
{
  return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p);
}

/*
 *  This function calculates the numerical flux using the Lax-Friedrichs flux.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveKernel<dim, Number>::calculate_lax_friedrichs_flux(scalar const & value_m,
                                                               scalar const & value_p,
                                                               scalar const & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);
  scalar jump_value    = value_m - value_p;
  scalar lambda        = std::abs(normal_velocity);

  return normal_velocity * average_value + 0.5 * lambda * jump_value;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveKernel<dim, Number>::calculate_lax_friedrichs_flux(
    scalar const & value_m,
    scalar const & value_p,
    scalar const & normal_velocity_m,
    scalar const & normal_velocity_p) const
{
  scalar jump_value = value_m - value_p;
  scalar lambda     = std::max(std::abs(normal_velocity_m), std::abs(normal_velocity_p));

  return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p) +
         0.5 * lambda * jump_value;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveKernel<dim, Number>::calculate_flux(unsigned int const q,
                                                IntegratorFace &   integrator,
                                                scalar const &     value_m,
                                                scalar const &     value_p,
                                                Number const &     time,
                                                bool const exterior_velocity_available) const
{
  vector normal = integrator.get_normal_vector(q);
  scalar flux   = make_vectorized_array<Number>(0.0);

  if(data.type_velocity_field == TypeVelocityField::Analytical)
  {
    Point<dim, scalar> q_points = integrator.quadrature_point(q);

    vector velocity = evaluate_vectorial_function(data.velocity, q_points, time);

    scalar normal_velocity = velocity * normal;

    if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
    {
      flux = calculate_central_flux(value_m, value_p, normal_velocity);
    }
    else if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
    {
      flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity);
    }
  }
  else if(data.type_velocity_field == TypeVelocityField::Numerical)
  {
    vector velocity_m = integrator_velocity_m->get_value(q);
    vector velocity_p =
      exterior_velocity_available ? integrator_velocity_p->get_value(q) : velocity_m;

    scalar normal_velocity_m = velocity_m * normal;
    scalar normal_velocity_p = velocity_p * normal;

    if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
    {
      flux = calculate_central_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
    }
    else if(data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
    {
      flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return flux;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  ConvectiveKernel<dim, Number>::get_volume_flux(IntegratorCell &   integrator,
                                                 unsigned int const q,
                                                 Number const &     time) const
{
  vector velocity;

  if(data.type_velocity_field == TypeVelocityField::Analytical)
  {
    velocity = evaluate_vectorial_function(data.velocity, integrator.quadrature_point(q), time);
  }
  else if(data.type_velocity_field == TypeVelocityField::Numerical)
  {
    velocity = integrator_velocity->get_value(q);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return (-integrator.get_value(q) * velocity);
}

template class ConvectiveKernel<2, float>;
template class ConvectiveKernel<2, double>;

template class ConvectiveKernel<3, float>;
template class ConvectiveKernel<3, double>;


} // namespace Operators

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ConvectiveOperatorData<dim> const & operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  kernel.reinit(matrix_free, operator_data.kernel_data, operator_data.quad_index);

  this->integrator_flags = kernel.get_integrator_flags();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity(VectorType const & velocity_in) const
{
  kernel.set_velocity(velocity_in);
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return kernel.get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  kernel.reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel.reinit_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel.reinit_boundary_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                        unsigned int const       face,
                                                        types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel.reinit_face_cell_based(cell, face, boundary_id);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_gradient(kernel.get_volume_flux(integrator, q, this->eval_time), q);
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

    scalar flux = kernel.calculate_flux(q, integrator_m, value_m, value_p, this->eval_time, true);

    integrator_m.submit_value(flux, q);
    integrator_p.submit_value(-flux, q);
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

    scalar flux = kernel.calculate_flux(q, integrator_m, value_m, value_p, this->eval_time, true);

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

    // TODO
    // The matrix-free implementation in deal.II does currently not allow to access neighboring data
    // in case of cell-based face loops. We therefore have to use integrator_velocity_m twice to
    // avoid the problem of accessing data of the neighboring element. Note that this variant
    // calculates the diagonal and block-diagonal only approximately. The theoretically correct
    // version using integrator_velocity_p is currently not implemented in deal.II.
    bool exterior_velocity_available = false; // TODO -> set to true once functionality is available
    scalar flux                      = kernel.calculate_flux(
      q, integrator_m, value_m, value_p, this->eval_time, exterior_velocity_available);

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

    scalar flux = kernel.calculate_flux(q, integrator_p, value_m, value_p, this->eval_time, true);

    // minus sign since n⁺ = -n⁻
    integrator_p.submit_value(-flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator_m,
                                                      OperatorType const &       operator_type,
                                                      types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, integrator_m, operator_type);

    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              integrator_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              this->operator_data.bc,
                                              this->eval_time);

    // In case of numerical velocity field:
    // Simply use velocity_p = velocity_m on boundary faces -> exterior_velocity_available = false.
    scalar flux = kernel.calculate_flux(q, integrator_m, value_m, value_p, this->eval_time, false);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  ConvectiveOperatorData<dim> const &  operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace ConvDiff
