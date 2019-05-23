#include "laplace_operator.h"

#include "../../functionalities/evaluate_functions.h"

namespace Poisson
{
template<int dim, typename Number>
LaplaceOperator<dim, Number>::LaplaceOperator()
  : OperatorBase<dim, Number, LaplaceOperatorData<dim>>()
{
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &   mf_data,
                                     AffineConstraints<double> const & constraint_matrix,
                                     LaplaceOperatorData<dim> const &  operator_data) const
{
  Base::reinit(mf_data, constraint_matrix, operator_data);
  // calculate penalty parameters
  MappingQGeneric<dim> mapping(operator_data.degree_mapping);
  IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                               *this->data,
                                               mapping,
                                               this->operator_data.degree,
                                               this->operator_data.dof_index);
}

template<int dim, typename Number>
bool
LaplaceOperator<dim, Number>::is_singular() const
{
  return this->operator_is_singular();
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, Number>::calculate_value_flux(scalar const & jump_value) const
{
  return -0.5 * jump_value;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, Number>::calculate_interior_value(unsigned int const   q,
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

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, Number>::calculate_exterior_value(scalar const &           value_m,
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

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, Number>::calculate_gradient_flux(scalar const & normal_gradient_m,
                                                        scalar const & normal_gradient_p,
                                                        scalar const & jump_value,
                                                        scalar const & penalty_parameter) const
{
  return 0.5 * (normal_gradient_m + normal_gradient_p) - penalty_parameter * jump_value;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, Number>::calculate_interior_normal_gradient(
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

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, Number>::calculate_exterior_normal_gradient(
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

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_cell_integral(FEEvalCell & fe_eval,
                                               unsigned int const /*cell*/) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_face_integral(FEEvalFace & fe_eval,
                                               FEEvalFace & fe_eval_neighbor,
                                               unsigned int const /*face*/) const
{
  scalar tau_IP =
    std::max(fe_eval.read_cell_data(array_penalty_parameter),
             fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
    IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);

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
    fe_eval_neighbor.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_face_int_integral(FEEvalFace & fe_eval,
                                                   FEEvalFace & fe_eval_neighbor,
                                                   unsigned int const /*face*/) const
{
  scalar tau_IP =
    std::max(fe_eval.read_cell_data(array_penalty_parameter),
             fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
    IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);

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

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_face_ext_integral(FEEvalFace & fe_eval,
                                                   FEEvalFace & fe_eval_neighbor,
                                                   unsigned int const /*face*/) const
{
  scalar tau_IP =
    std::max(fe_eval.read_cell_data(array_penalty_parameter),
             fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
    IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);

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

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_boundary_integral(FEEvalFace &               fe_eval,
                                                   OperatorType const &       operator_type,
                                                   types::boundary_id const & boundary_id,
                                                   unsigned int const /*face*/) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  scalar tau_IP =
    fe_eval.read_cell_data(array_penalty_parameter) *
    IP::get_penalty_factor<Number>(this->operator_data.degree, this->operator_data.IP_factor);

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

template<int dim, typename Number>
void
LaplaceOperator<dim, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  LaplaceOperatorData<dim> const &     operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  unsigned int counter = 0;
  if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
    counter++;

  if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
    counter++;

  if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
    counter++;

  AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
}

template class LaplaceOperator<2, float>;
template class LaplaceOperator<2, double>;

template class LaplaceOperator<3, float>;
template class LaplaceOperator<3, double>;

} // namespace Poisson
