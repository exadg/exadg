#include "convective_operator.h"

#include "verify_boundary_conditions.h"
#include "weak_boundary_conditions.h"

namespace ConvDiff
{
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &     matrix_free,
                                        AffineConstraints<double> const &   constraint_matrix,
                                        ConvectiveOperatorData<dim> const & operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  if(operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    matrix_free.initialize_dof_vector(velocity, operator_data.dof_index_velocity);

    fe_eval_velocity.reset(new FEEvalCellVelocity(matrix_free,
                                                  operator_data.dof_index_velocity,
                                                  operator_data.quad_index));

    fe_eval_velocity_m.reset(new FEEvalFaceVelocity(
      matrix_free, true, operator_data.dof_index_velocity, operator_data.quad_index));

    fe_eval_velocity_p.reset(new FEEvalFaceVelocity(
      matrix_free, false, operator_data.dof_index_velocity, operator_data.quad_index));
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity(VectorType const & velocity_in) const
{
  AssertThrow(this->operator_data.type_velocity_field == TypeVelocityField::Numerical,
              ExcMessage("Invalid parameter type_velocity_field."));

  velocity = velocity_in;

  velocity.update_ghost_values();
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return velocity;
}

/*
 *  This function calculates the numerical flux using the central flux.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, Number>::calculate_central_flux(scalar const & value_m,
                                                          scalar const & value_p,
                                                          scalar const & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);

  return normal_velocity * average_value;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, Number>::calculate_central_flux(scalar const & value_m,
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
  ConvectiveOperator<dim, Number>::calculate_lax_friedrichs_flux(
    scalar const & value_m,
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
  ConvectiveOperator<dim, Number>::calculate_lax_friedrichs_flux(
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
  ConvectiveOperator<dim, Number>::calculate_flux(unsigned int const q,
                                                  FEEvalFace &       fe_eval,
                                                  scalar const &     value_m,
                                                  scalar const &     value_p,
                                                  bool const exterior_velocity_available) const
{
  vector normal = fe_eval.get_normal_vector(q);
  scalar flux   = make_vectorized_array<Number>(0.0);

  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

    vector velocity =
      evaluate_vectorial_function(this->operator_data.velocity, q_points, this->eval_time);

    scalar normal_velocity = velocity * normal;

    if(this->operator_data.numerical_flux_formulation ==
       NumericalFluxConvectiveOperator::CentralFlux)
    {
      flux = calculate_central_flux(value_m, value_p, normal_velocity);
    }
    else if(this->operator_data.numerical_flux_formulation ==
            NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
    {
      flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity);
    }
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    vector velocity_m = fe_eval_velocity_m->get_value(q);
    vector velocity_p = exterior_velocity_available ? fe_eval_velocity_p->get_value(q) : velocity_m;

    scalar normal_velocity_m = velocity_m * normal;
    scalar normal_velocity_p = velocity_p * normal;

    if(this->operator_data.numerical_flux_formulation ==
       NumericalFluxConvectiveOperator::CentralFlux)
    {
      flux = calculate_central_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
    }
    else if(this->operator_data.numerical_flux_formulation ==
            NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
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
  ConvectiveOperator<dim, Number>::get_volume_flux(FEEvalCell & fe_eval, unsigned int const q) const
{
  vector velocity;

  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    velocity = evaluate_vectorial_function(this->operator_data.velocity,
                                           fe_eval.quadrature_point(q),
                                           this->eval_time);
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    velocity = fe_eval_velocity->get_value(q);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return (-fe_eval.get_value(q) * velocity);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity->reinit(cell);
    fe_eval_velocity->gather_evaluate(velocity, true, false, false);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    fe_eval_velocity_p->reinit(face);
    fe_eval_velocity_p->gather_evaluate(velocity, true, false);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                        unsigned int const       face,
                                                        types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(cell, face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      // TODO: Matrix-free implementation in deal.II does currently not allow to access data of the
      // neighboring element in case of cell-based face loops.
      //      fe_eval_velocity_p->reinit(cell, face);
      //      fe_eval_velocity_p->gather_evaluate(velocity, true, false);
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(FEEvalCell & fe_eval) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    fe_eval.submit_gradient(get_volume_flux(fe_eval, q), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral(FEEvalFace & fe_eval_m,
                                                  FEEvalFace & fe_eval_p) const
{
  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    scalar value_m = fe_eval_m.get_value(q);
    scalar value_p = fe_eval_p.get_value(q);

    scalar flux = calculate_flux(q, fe_eval_m, value_m, value_p, true);

    fe_eval_m.submit_value(flux, q);
    fe_eval_p.submit_value(-flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral(FEEvalFace & fe_eval_m,
                                                      FEEvalFace & fe_eval_p) const
{
  (void)fe_eval_p;

  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);
    scalar value_m = fe_eval_m.get_value(q);

    scalar flux = calculate_flux(q, fe_eval_m, value_m, value_p, true);

    fe_eval_m.submit_value(flux, q);
  }
}

// TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
// cell-based face loops
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral_cell_based(FEEvalFace & fe_eval_m,
                                                                 FEEvalFace & fe_eval_p) const
{
  (void)fe_eval_p;

  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);
    scalar value_m = fe_eval_m.get_value(q);

    // TODO
    // The matrix-free implementation in deal.II does currently not allow to access neighboring data
    // in case of cell-based face loops. We therefore have to use fe_eval_velocity_m twice to avoid
    // the problem of accessing data of the neighboring element. Note that this variant calculates
    // the diagonal and block-diagonal only approximately. The theoretically correct version using
    // fe_eval_velocity_p is currently not implemented in deal.II.
    bool exterior_velocity_available = false; // TODO -> set to true once functionality is available
    scalar flux = calculate_flux(q, fe_eval_m, value_m, value_p, exterior_velocity_available);

    fe_eval_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_ext_integral(FEEvalFace & fe_eval_m,
                                                      FEEvalFace & fe_eval_p) const
{
  (void)fe_eval_m;

  for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m = make_vectorized_array<Number>(0.0);
    scalar value_p = fe_eval_p.get_value(q);

    scalar flux = calculate_flux(q, fe_eval_p, value_m, value_p, true);

    // minus sign since n⁺ = -n⁻
    fe_eval_p.submit_value(-flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral(FEEvalFace &               fe_eval_m,
                                                      OperatorType const &       operator_type,
                                                      types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, fe_eval_m, operator_type);

    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              fe_eval_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              this->operator_data.bc,
                                              this->eval_time);

    // In case of numerical velocity field:
    // Simply use velocity_p = velocity_m for boundary integrals -> exterior_velocity_available =
    // false.
    scalar flux = calculate_flux(q, fe_eval_m, value_m, value_p, false);

    fe_eval_m.submit_value(flux, q);
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
