#include "diffusive_operator.h"

#include "verify_boundary_conditions.h"

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/interior_penalty_parameter.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
DiffusiveOperator<dim, degree, Number>::DiffusiveOperator() : diffusivity(-1.0)
{
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::reinit(
  MatrixFree<dim, Number> const &    mf_data,
  AffineConstraints<double> const &  constraint_matrix,
  DiffusiveOperatorData<dim> const & operator_data) const
{
  // TODO: adjust as in the case of laplace
  MappingQGeneric<dim> mapping(degree);
  this->reinit(mapping, mf_data, constraint_matrix, operator_data);
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::reinit(
  Mapping<dim> const &               mapping,
  MatrixFree<dim, Number> const &    mf_data,
  AffineConstraints<double> const &  constraint_matrix,
  DiffusiveOperatorData<dim> const & operator_data) const
{
  Base::reinit(mf_data, constraint_matrix, operator_data);

  IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                       *this->data,
                                                       mapping,
                                                       this->operator_data.dof_index);

  diffusivity = this->operator_data.diffusivity;
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  AssertThrow(diffusivity > 0.0, ExcMessage("Diffusivity is not set!"));
  Base::apply_add(dst, src);
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::apply_add(VectorType & /*dst*/,
                                                  VectorType const & /*src*/,
                                                  Number const /*time*/) const
{
  // This function has to be overwritten explicitly. Otherwise the compiler
  // complains that this function of the base class is hidden by the other apply_add
  AssertThrow(false,
              ExcMessage("DiffusiveOperator cannot be called with additional parameter time!"));
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, degree, Number>::calculate_value_flux(scalar const & jump_value) const
{
  return -0.5 * diffusivity * jump_value;
}

/*
 *  The following two functions calculate the interior_value/exterior_value
 *  depending on the operator type, the type of the boundary face
 *  and the given boundary conditions.
 *
 *                            +-----------------------------+-----------------------+
 *                            | Dirichlet boundaries        | Neumann boundaries    |
 *  +-------------------------+-----------------------------+-----------------------+
 *  | full operator           | phi⁺ = -phi⁻ + 2g           | phi⁺ = phi⁻           |
 *  +-------------------------+-----------------------------+-----------------------+
 *  | homogeneous operator    | phi⁺ = -phi⁻                | phi⁺ = phi⁻           |
 *  +-------------------------+-----------------------------+-----------------------+
 *  | inhomogeneous operator  | phi⁺ = -phi⁻ + 2g, phi⁻ = 0 | phi⁺ = phi⁻, phi⁻ = 0 |
 *  +-------------------------+-----------------------------+-----------------------+
 */
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, degree, Number>::calculate_interior_value(
    unsigned int const   q,
    FEEvalFace const &   fe_eval,
    OperatorType const & operator_type) const
{
  scalar value_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    value_m = fe_eval.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    value_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return value_m;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, degree, Number>::calculate_exterior_value(
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

/*
 *  Calculation of gradient flux. Strictly speaking, this value is not a
 * numerical flux since
 *  the flux is multiplied by the normal vector, i.e., "gradient_flux" =
 * numerical_flux * normal,
 *  where normal denotes the normal vector of element e⁻.
 */
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, degree, Number>::calculate_gradient_flux(
    scalar const & normal_gradient_m,
    scalar const & normal_gradient_p,
    scalar const & jump_value,
    scalar const & penalty_parameter) const
{
  return diffusivity * 0.5 * (normal_gradient_m + normal_gradient_p) -
         diffusivity * penalty_parameter * jump_value;
}

// clang-format off
  /*
   *  The following two functions calculate the interior/exterior velocity gradient
   *  in normal direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n + 2h                    |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n                         |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁺)*n = grad(phi⁻)*n, grad(phi⁻)*n = 0 | grad(phi⁺)*n = -grad(phi⁻)*n + 2h, grad(phi⁻)*n  = 0 |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = 0                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                           | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   */
// clang-format on
template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, degree, Number>::calculate_interior_normal_gradient(
    unsigned int const   q,
    FEEvalFace const &   fe_eval,
    OperatorType const & operator_type) const
{
  scalar normal_gradient_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = fe_eval.get_normal_derivative(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    normal_gradient_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return normal_gradient_m;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  DiffusiveOperator<dim, degree, Number>::calculate_exterior_normal_gradient(
    scalar const &           normal_gradient_m,
    unsigned int const       q,
    FEEvalFace const &       fe_eval,
    OperatorType const &     operator_type,
    BoundaryType const &     boundary_type,
    types::boundary_id const boundary_id) const
{
  scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        this->operator_data.bc->neumann_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar h = evaluate_scalar_function(it->second, q_points, this->eval_time);

      normal_gradient_p = -normal_gradient_m + 2.0 * h;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      normal_gradient_p = -normal_gradient_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::do_cell_integral(FEEvalCell & fe_eval,
                                                         unsigned int const /*cell*/) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_gradient(fe_eval.get_gradient(q) * diffusivity, q);
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::do_face_integral(FEEvalFace & fe_eval,
                                                         FEEvalFace & fe_eval_neighbor,
                                                         unsigned int const /*face*/) const
{
  scalar tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                           fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    scalar jump_value = fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
    scalar value_flux = calculate_value_flux(jump_value);

    scalar normal_gradient_m = fe_eval.get_normal_derivative(q);
    scalar normal_gradient_p = fe_eval_neighbor.get_normal_derivative(q);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_derivative(value_flux, q);
    fe_eval_neighbor.submit_normal_derivative(value_flux, q);

    fe_eval.submit_value(-gradient_flux, q);
    // + sign since n⁺ = -n⁻
    fe_eval_neighbor.submit_value(gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::do_face_int_integral(FEEvalFace & fe_eval,
                                                             FEEvalFace & fe_eval_neighbor,
                                                             unsigned int const /*face*/) const
{
  scalar tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                           fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    // set exterior value to zero
    scalar jump_value = fe_eval.get_value(q);
    scalar value_flux = calculate_value_flux(jump_value);

    // set exterior value to zero
    scalar normal_gradient_m = fe_eval.get_normal_derivative(q);
    scalar normal_gradient_p = make_vectorized_array<Number>(0.0);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_derivative(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::do_face_ext_integral(FEEvalFace & fe_eval,
                                                             FEEvalFace & fe_eval_neighbor,
                                                             unsigned int const /*face*/) const
{
  scalar tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                           fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    // set value_m to zero
    scalar jump_value = fe_eval_neighbor.get_value(q);
    scalar value_flux = calculate_value_flux(jump_value);

    // set gradient_m to zero
    scalar normal_gradient_m = make_vectorized_array<Number>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    scalar normal_gradient_p = -fe_eval_neighbor.get_normal_derivative(q);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    // minus sign since n⁺ = -n⁻
    fe_eval_neighbor.submit_normal_derivative(-value_flux, q);
    fe_eval_neighbor.submit_value(-gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::do_boundary_integral(FEEvalFace &         fe_eval,
                                                             OperatorType const & operator_type,
                                                             types::boundary_id const & boundary_id,
                                                             unsigned int const /*face*/) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  scalar tau_IP = fe_eval.read_cell_data(array_penalty_parameter) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, fe_eval, operator_type);
    scalar value_p =
      calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    scalar jump_value = value_m - value_p;
    scalar value_flux = calculate_value_flux(jump_value);

    scalar normal_gradient_m = calculate_interior_normal_gradient(q, fe_eval, operator_type);
    scalar normal_gradient_p = calculate_exterior_normal_gradient(
      normal_gradient_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_derivative(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
DiffusiveOperator<dim, degree, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  DiffusiveOperatorData<dim> const &   operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  ConvDiff::do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

} // namespace ConvDiff

#include "diffusive_operator.hpp"
