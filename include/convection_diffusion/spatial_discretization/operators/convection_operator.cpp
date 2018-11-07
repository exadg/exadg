#include "convection_operator.h"

#include "verify_boundary_conditions.h"

#include "../../../functionalities/evaluate_functions.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::initialize(
  MatrixFree<dim, Number> const &     mf_data,
  ConvectiveOperatorData<dim> const & operator_data_in,
  unsigned int                        level_mg_handler)
{
  AffineConstraints<double> constraint_matrix;
  Parent::reinit(mf_data, constraint_matrix, operator_data_in, level_mg_handler);
}

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::initialize(
  MatrixFree<dim, Number> const &     mf_data,
  AffineConstraints<double> const &   constraint_matrix,
  ConvectiveOperatorData<dim> const & operator_data_in,
  unsigned int                        level_mg_handler)
{
  Parent::reinit(mf_data, constraint_matrix, operator_data_in, level_mg_handler);
}

/*
 *  This function calculates the numerical flux using the central flux.
 */
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, Number>::calculate_central_flux(scalar & value_m,
                                                                  scalar & value_p,
                                                                  scalar & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);

  return normal_velocity * average_value;
}

/*
 *  This function calculates the numerical flux using the Lax-Friedrichs flux.
 */
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, Number>::calculate_lax_friedrichs_flux(
    scalar & value_m,
    scalar & value_p,
    scalar & normal_velocity) const
{
  scalar average_value = 0.5 * (value_m + value_p);
  scalar jump_value    = value_m - value_p;
  scalar lambda        = std::abs(normal_velocity);

  return normal_velocity * average_value + 0.5 * lambda * jump_value;
}

/*
 * This function calculates the numerical flux where the type of the numerical flux depends on the
 * specified input parameter.
 */
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, Number>::calculate_flux(unsigned int const q,
                                                          FEEvalFace &       fe_eval,
                                                          scalar &           value_m,
                                                          scalar &           value_p) const
{
  scalar flux = make_vectorized_array<Number>(0.0);

  Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

  vector velocity;

  evaluate_vectorial_function(velocity, this->operator_data.velocity, q_points, this->eval_time);

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
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, Number>::calculate_interior_value(
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

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  ConvectiveOperator<dim, degree, Number>::calculate_exterior_value(
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
      scalar g = make_vectorized_array<Number>(0.0);
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it                          = this->operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);
      evaluate_scalar_function(g, it->second, q_points, this->eval_time);

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

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::do_cell_integral(FEEvalCell & fe_eval) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

    vector velocity;

    evaluate_vectorial_function(velocity, this->operator_data.velocity, q_points, this->eval_time);

    fe_eval.submit_gradient(-fe_eval.get_value(q) * velocity, q);
  }
}

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::do_face_integral(FEEvalFace & fe_eval,
                                                          FEEvalFace & fe_eval_neighbor) const
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

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::do_face_int_integral(
  FEEvalFace & fe_eval,
  FEEvalFace & /*fe_eval_neighbor*/) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    scalar value_m = fe_eval.get_value(q);
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);

    scalar numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

    fe_eval.submit_value(numerical_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::do_face_ext_integral(FEEvalFace & /*fe_eval*/,
                                                              FEEvalFace & fe_eval_neighbor) const
{
  for(unsigned int q = 0; q < fe_eval_neighbor.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m        = make_vectorized_array<Number>(0.0);
    scalar value_p        = fe_eval_neighbor.get_value(q);
    scalar numerical_flux = calculate_flux(q, fe_eval_neighbor, value_m, value_p);

    // hack (minus sign) since n⁺ = -n⁻
    fe_eval_neighbor.submit_value(-numerical_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::do_boundary_integral(
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
    scalar numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

    fe_eval.submit_value(numerical_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
ConvectiveOperator<dim, degree, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  ConvectiveOperatorData<dim> const &  operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  ConvDiff::do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

} // namespace ConvDiff

#include "convection_operator.hpp"
