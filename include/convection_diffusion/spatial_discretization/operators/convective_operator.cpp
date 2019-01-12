#include "convective_operator.h"

#include "verify_boundary_conditions.h"

#include "../../../functionalities/evaluate_functions.h"

namespace ConvDiff
{
template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::reinit(
  MatrixFree<dim, Number> const &     data,
  AffineConstraints<double> const &   constraint_matrix,
  ConvectiveOperatorData<dim> const & operator_data)
{
  Base::reinit(data, constraint_matrix, operator_data);

  if(operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    data.initialize_dof_vector(velocity, operator_data.dof_index_velocity);

    fe_eval_velocity.reset(
      new FEEvalCellVelocity(data, operator_data.dof_index_velocity, operator_data.quad_index));

    fe_eval_velocity_m.reset(new FEEvalFaceVelocity(
      data, true, operator_data.dof_index_velocity, operator_data.quad_index));

    fe_eval_velocity_p.reset(new FEEvalFaceVelocity(
      data, false, operator_data.dof_index_velocity, operator_data.quad_index));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::set_velocity(
  VectorType const & velocity_in) const
{
  AssertThrow(this->operator_data.type_velocity_field == TypeVelocityField::Numerical,
              ExcMessage("Invalid parameter type_velocity_field."));

  velocity = velocity_in;

  velocity.update_ghost_values();
}

template<int dim, int degree, int degree_velocity, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, degree, degree_velocity, Number>::get_velocity() const
{
  return velocity;
}

/*
 *  This function calculates the numerical flux using the central flux.
 */
template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_central_flux(
    scalar const & value_m,
    scalar const & value_p,
    scalar const & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);

  return normal_velocity * average_value;
}

template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_central_flux(
    scalar const & value_m,
    scalar const & value_p,
    scalar const & normal_velocity_m,
    scalar const & normal_velocity_p) const
{
  return 0.5 * (normal_velocity_m * value_m + normal_velocity_p * value_p);
}

/*
 *  This function calculates the numerical flux using the Lax-Friedrichs flux.
 */
template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_lax_friedrichs_flux(
    scalar const & value_m,
    scalar const & value_p,
    scalar const & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);
  scalar jump_value    = value_m - value_p;
  scalar lambda        = std::abs(normal_velocity);

  return normal_velocity * average_value + 0.5 * lambda * jump_value;
}

template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_lax_friedrichs_flux(
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

/*
 * This function calculates the numerical flux where the type of the numerical flux depends on the
 * specified input parameter.
 */
template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_flux(
    unsigned int const q,
    FEEvalFace &       fe_eval,
    scalar const &     value_m,
    scalar const &     value_p) const
{
  scalar flux = make_vectorized_array<Number>(0.0);

  Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

  vector velocity =
    evaluate_vectorial_function(this->operator_data.velocity, q_points, this->eval_time);

  vector normal = fe_eval.get_normal_vector(q);

  scalar normal_velocity = velocity * normal;

  if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
  {
    flux = calculate_central_flux(value_m, value_p, normal_velocity);
  }
  else if(this->operator_data.numerical_flux_formulation ==
          NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
  {
    flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity);
  }

  return flux;
}

template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_flux(
    scalar const & value_m,
    scalar const & value_p,
    scalar const & normal_velocity_m,
    scalar const & normal_velocity_p) const
{
  scalar flux = make_vectorized_array<Number>(0.0);

  if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
  {
    flux = calculate_central_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
  }
  else if(this->operator_data.numerical_flux_formulation ==
          NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
  {
    flux = calculate_lax_friedrichs_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);
  }

  return flux;
}

/*
 *  The following two functions calculate the interior_value/exterior_value
 *  depending on the operator type, the type of the boundary face
 *  and the given boundary conditions.
 *
 *                            +----------------------+--------------------+
 *                            | Dirichlet boundaries | Neumann boundaries |
 *  +-------------------------+----------------------+--------------------+
 *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
 *  +-------------------------+----------------------+--------------------+
 *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
 *  +-------------------------+----------------------+--------------------+
 *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
 *  +-------------------------+----------------------+--------------------+
 */
template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_interior_value(
    unsigned int const   q,
    FEEvalFace const &   fe_eval,
    OperatorType const & operator_type) const
{
  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    return fe_eval.get_value(q);
  else if(operator_type == OperatorType::inhomogeneous)
    return make_vectorized_array<Number>(0.0);
  else
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));

  return make_vectorized_array<Number>(0.0);
}

template<int dim, int degree, int degree_velocity, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, degree_velocity, Number>::calculate_exterior_value(
    scalar const &           value_m,
    unsigned int const       q,
    FEEvalFace const &       fe_eval,
    OperatorType const &     operator_type,
    BoundaryType const &     boundary_type,
    types::boundary_id const boundary_id) const
{
  scalar value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        this->operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar g = evaluate_scalar_function(it->second, q_points, this->eval_time);

      value_p = -value_m + 2.0 * g;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      value_p = -value_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_cell_integral(
  FEEvalCell &       fe_eval,
  unsigned int const cell) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      vector velocity =
        evaluate_vectorial_function(this->operator_data.velocity, q_points, this->eval_time);

      fe_eval.submit_gradient(-fe_eval.get_value(q) * velocity, q);
    }
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity->reinit(cell);
    fe_eval_velocity->gather_evaluate(velocity, true, false, false);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_gradient(-fe_eval.get_value(q) * fe_eval_velocity->get_value(q), q);
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_face_integral(
  FEEvalFace &       fe_eval,
  FEEvalFace &       fe_eval_neighbor,
  unsigned int const face) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar value_m = fe_eval.get_value(q);
      scalar value_p = fe_eval_neighbor.get_value(q);

      scalar numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

      fe_eval.submit_value(numerical_flux, q);
      fe_eval_neighbor.submit_value(-numerical_flux, q);
    }
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    fe_eval_velocity_p->reinit(face);
    fe_eval_velocity_p->gather_evaluate(velocity, true, false);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar value_m = fe_eval.get_value(q);
      scalar value_p = fe_eval_neighbor.get_value(q);

      vector velocity_m = fe_eval_velocity_m->get_value(q);
      vector velocity_p = fe_eval_velocity_p->get_value(q);

      vector normal = fe_eval.get_normal_vector(q);

      scalar normal_velocity_m = velocity_m * normal;
      scalar normal_velocity_p = velocity_p * normal;

      scalar flux = calculate_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);

      fe_eval.submit_value(flux, q);
      fe_eval_neighbor.submit_value(-flux, q);
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_face_int_integral(
  FEEvalFace & fe_eval_m,
  FEEvalFace & /*fe_eval_p*/,
  unsigned int const face) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      scalar value_m = fe_eval_m.get_value(q);
      // set value_p to zero
      scalar value_p = make_vectorized_array<Number>(0.0);

      scalar numerical_flux = calculate_flux(q, fe_eval_m, value_m, value_p);

      fe_eval_m.submit_value(numerical_flux, q);
    }
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    fe_eval_velocity_p->reinit(face);
    fe_eval_velocity_p->gather_evaluate(velocity, true, false);

    do_face_int_integral(fe_eval_m, *fe_eval_velocity_m, *fe_eval_velocity_p);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_face_int_integral_cell_based(
  FEEvalFace &       fe_eval_m,
  FEEvalFace &       fe_eval_p,
  unsigned int const cell,
  unsigned int const face) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    this->do_face_int_integral(fe_eval_m, fe_eval_p, face);
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    std::cout << "ConvectiveOperator::do_face_int_integral_cell_based" << std::endl;
    fe_eval_velocity_m->reinit(cell, face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    // TODO: Matrix-free implementation in deal.II does currently not allow to access data of the
    // neighboring element.
    //  fe_eval_velocity_p->reinit(cell, face);
    //  fe_eval_velocity_p->gather_evaluate(velocity, true, false);

    //  do_face_int_integral(fe_eval_m, *fe_eval_velocity_m, *fe_eval_velocity_p);

    // TODO: we have to use fe_eval_velocity_m twice to avoid the above problem
    do_face_int_integral(fe_eval_m, *fe_eval_velocity_m, *fe_eval_velocity_m);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_face_int_integral(
  FEEvalFace &         fe_eval_m,
  FEEvalFaceVelocity & fe_eval_velocity_m,
  FEEvalFaceVelocity & fe_eval_velocity_p) const
{
  for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
  {
    scalar value_m = fe_eval_m.get_value(q);
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);

    vector velocity_m = fe_eval_velocity_m.get_value(q);
    vector velocity_p = fe_eval_velocity_p.get_value(q);

    vector normal = fe_eval_m.get_normal_vector(q);

    scalar normal_velocity_m = velocity_m * normal;
    scalar normal_velocity_p = velocity_p * normal;

    scalar flux = calculate_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);

    fe_eval_m.submit_value(flux, q);
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_face_ext_integral(
  FEEvalFace & /*fe_eval_m*/,
  FEEvalFace &       fe_eval_p,
  unsigned int const face) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
    {
      // set value_m to zero
      scalar value_m        = make_vectorized_array<Number>(0.0);
      scalar value_p        = fe_eval_p.get_value(q);
      scalar numerical_flux = calculate_flux(q, fe_eval_p, value_m, value_p);

      // hack (minus sign) since n⁺ = -n⁻
      fe_eval_p.submit_value(-numerical_flux, q);
    }
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    fe_eval_velocity_p->reinit(face);
    fe_eval_velocity_p->gather_evaluate(velocity, true, false);

    for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
    {
      // set value_m to zero
      scalar value_m = make_vectorized_array<Number>(0.0);
      scalar value_p = fe_eval_p.get_value(q);

      vector velocity_m = fe_eval_velocity_m->get_value(q);
      vector velocity_p = fe_eval_velocity_p->get_value(q);

      vector normal = fe_eval_p.get_normal_vector(q);

      scalar normal_velocity_m = velocity_m * normal;
      scalar normal_velocity_p = velocity_p * normal;

      scalar flux = calculate_flux(value_m, value_p, normal_velocity_m, normal_velocity_p);

      // minus sign since n⁺ = -n⁻
      fe_eval_p.submit_value(-flux, q);
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_boundary_integral(
  FEEvalFace &               fe_eval,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id,
  unsigned int const         face) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
  {
    BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar value_m = calculate_interior_value(q, fe_eval, operator_type);
      scalar value_p =
        calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);
      scalar numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

      fe_eval.submit_value(numerical_flux, q);
    }
  }
  else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);

    do_boundary_integral(fe_eval, operator_type, boundary_id);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_boundary_integral_cell_based(
  FEEvalFace &               fe_eval,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id,
  unsigned int const         cell,
  unsigned int const         face) const
{
  if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_eval_velocity_m->reinit(cell, face);
    fe_eval_velocity_m->gather_evaluate(velocity, true, false);
  }

  do_boundary_integral(fe_eval, operator_type, boundary_id);
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_boundary_integral(
  FEEvalFace &               fe_eval,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, fe_eval, operator_type);
    scalar value_p =
      calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);

    vector velocity_m;

    if(this->operator_data.type_velocity_field == TypeVelocityField::Analytical)
    {
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);
      velocity_m =
        evaluate_vectorial_function(this->operator_data.velocity, q_points, this->eval_time);
    }
    else if(this->operator_data.type_velocity_field == TypeVelocityField::Numerical)
    {
      velocity_m = fe_eval_velocity_m->get_value(q);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    vector normal = fe_eval.get_normal_vector(q);

    scalar normal_velocity_m = velocity_m * normal;

    scalar flux = calculate_flux(value_m, value_p, normal_velocity_m, normal_velocity_m);

    fe_eval.submit_value(flux, q);
  }
}

template<int dim, int degree, int degree_velocity, typename Number>
void
ConvectiveOperator<dim, degree, degree_velocity, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  ConvectiveOperatorData<dim> const &  operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  ConvDiff::do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

} // namespace ConvDiff

#include "convective_operator.hpp"
